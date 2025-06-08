import hashlib
import json
from typing import Optional, Any, Dict, List, AsyncIterator
import redis.asyncio as redis
from .logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages LLM response caching with intelligent key generation"""

    def __init__(
        self, redis_client: redis.Redis, ttl: int = 604800, namespace: str = "llmproxy"
    ):
        self.redis = redis_client
        self.ttl = ttl
        self.namespace = namespace
        self._hits = 0
        self._misses = 0
        self._streaming_hits = 0
        self._streaming_misses = 0

    def _generate_cache_key(self, request_data: dict) -> str:
        """Generate cache key from request parameters"""
        # Extract relevant fields for caching
        cache_data = {
            "model": request_data.get("model"),
            "messages": request_data.get("messages"),
            "temperature": request_data.get("temperature", 1.0),
            "max_tokens": request_data.get("max_tokens"),
            "top_p": request_data.get("top_p", 1.0),
            "frequency_penalty": request_data.get("frequency_penalty", 0),
            "presence_penalty": request_data.get("presence_penalty", 0),
            "tools": request_data.get("tools"),
            "tool_choice": request_data.get("tool_choice"),
            "seed": request_data.get("seed"),
        }

        # Remove None values
        cache_data = {k: v for k, v in cache_data.items() if v is not None}

        # Create deterministic hash
        cache_str = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()

        return f"{self.namespace}:{cache_hash}"

    def _should_cache(self, request_data: dict, ignore_streaming: bool = False) -> bool:
        """Determine if request should be cached"""
        # Check for cache control directives
        # Support both direct cache parameter and extra_body.cache
        cache_control = request_data.get("cache")
        if not cache_control:
            # Check extra_body for cache control
            extra_body = request_data.get("extra_body", {})
            cache_control = extra_body.get("cache")
        
        # If cache control is specified and no-cache is True, don't cache
        if cache_control and cache_control.get("no-cache", False):
            return False

        # Don't check streaming if explicitly ignored (for streaming cache implementation)
        if not ignore_streaming and request_data.get("stream", False):
            # For backward compatibility, we check if streaming caching is explicitly enabled
            # via cache control directive
            if not (cache_control and cache_control.get("stream-cache", False)):
                return False

        return True

    async def get(self, request_data: dict) -> Optional[dict]:
        """Get cached response for non-streaming requests"""
        if not self._should_cache(request_data):
            return None

        key = self._generate_cache_key(request_data)

        try:
            data = await self.redis.get(key)

            if data:
                self._hits += 1
                logger.info("cache_hit", key=key)
                return json.loads(data)

            self._misses += 1
            logger.debug("cache_miss", key=key)
            return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e), key=key)
            return None

    async def set(self, request_data: dict, response_data: dict):
        """Cache response for non-streaming requests"""
        if not self._should_cache(request_data):
            return

        # Don't cache error responses
        if "error" in response_data:
            return

        key = self._generate_cache_key(request_data)

        try:
            await self.redis.setex(key, self.ttl, json.dumps(response_data))
            logger.debug("cache_set", key=key, ttl=self.ttl)

        except Exception as e:
            logger.error("cache_set_error", error=str(e), key=key)

    async def get_streaming(self, request_data: dict) -> Optional[List[str]]:
        """Get cached streaming response chunks"""
        if not self._should_cache(request_data, ignore_streaming=True):
            return None

        key = f"{self._generate_cache_key(request_data)}:stream"

        try:
            # Get all chunks from Redis list
            chunks = await self.redis.lrange(key, 0, -1)
            
            if chunks:
                self._streaming_hits += 1
                logger.info("streaming_cache_hit", key=key, num_chunks=len(chunks))
                return chunks
            
            self._streaming_misses += 1
            logger.debug("streaming_cache_miss", key=key)
            return None

        except Exception as e:
            logger.error("streaming_cache_get_error", error=str(e), key=key)
            return None

    async def set_streaming_chunk(self, request_data: dict, chunk: str, finalize: bool = False):
        """Cache a streaming response chunk"""
        if not self._should_cache(request_data, ignore_streaming=True):
            return

        key = f"{self._generate_cache_key(request_data)}:stream"

        try:
            # Push chunk to Redis list
            await self.redis.rpush(key, chunk)
            
            # Set TTL when finalizing (after all chunks are stored)
            if finalize:
                await self.redis.expire(key, self.ttl)
                logger.debug("streaming_cache_finalized", key=key, ttl=self.ttl)

        except Exception as e:
            logger.error("streaming_cache_set_error", error=str(e), key=key, chunk_preview=chunk[:100])

    async def create_streaming_cache_writer(self, request_data: dict) -> "StreamingCacheWriter":
        """Create a streaming cache writer that intercepts and caches chunks"""
        return StreamingCacheWriter(self, request_data)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        streaming_total = self._streaming_hits + self._streaming_misses
        streaming_hit_rate = (self._streaming_hits / streaming_total * 100) if streaming_total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
            "streaming_hits": self._streaming_hits,
            "streaming_misses": self._streaming_misses,
            "streaming_total": streaming_total,
            "streaming_hit_rate": streaming_hit_rate,
        }

    async def invalidate_all(self) -> int:
        """Invalidate all cached entries for this namespace"""
        try:
            pattern = f"{self.namespace}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info("cache_invalidate_all", 
                           namespace=self.namespace, 
                           keys_deleted=deleted)
                return deleted
            else:
                logger.info("cache_invalidate_all_empty", namespace=self.namespace)
                return 0
                
        except Exception as e:
            logger.error("cache_invalidate_all_error", 
                        error=str(e), 
                        namespace=self.namespace)
            return 0

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cached entries matching a pattern"""
        try:
            # Ensure pattern includes namespace
            if not pattern.startswith(f"{self.namespace}:"):
                pattern = f"{self.namespace}:{pattern}"
                
            keys = await self.redis.keys(pattern)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info("cache_invalidate_pattern", 
                           pattern=pattern, 
                           keys_deleted=deleted)
                return deleted
            else:
                logger.info("cache_invalidate_pattern_empty", pattern=pattern)
                return 0
                
        except Exception as e:
            logger.error("cache_invalidate_pattern_error", 
                        error=str(e), 
                        pattern=pattern)
            return 0

    async def invalidate_request(self, request_data: dict) -> bool:
        """Invalidate cache for a specific request"""
        try:
            key = self._generate_cache_key(request_data)
            
            # Delete both regular and streaming cache entries
            regular_key = key
            streaming_key = f"{key}:stream"
            
            deleted = 0
            if await self.redis.exists(regular_key):
                deleted += await self.redis.delete(regular_key)
            if await self.redis.exists(streaming_key):
                deleted += await self.redis.delete(streaming_key)
            
            if deleted > 0:
                logger.info("cache_invalidate_request", 
                           key=key, 
                           entries_deleted=deleted)
                return True
            else:
                logger.debug("cache_invalidate_request_not_found", key=key)
                return False
                
        except Exception as e:
            logger.error("cache_invalidate_request_error", 
                        error=str(e), 
                        key=key if 'key' in locals() else 'unknown')
            return False


class StreamingCacheWriter:
    """Helper class to intercept and cache streaming chunks"""
    
    def __init__(self, cache_manager: CacheManager, request_data: dict):
        self.cache_manager = cache_manager
        self.request_data = request_data
        self.chunks_written = 0
        self._error_occurred = False
    
    async def write_and_yield(self, chunk: str) -> str:
        """Write chunk to cache and yield it"""
        # Don't cache if error occurred
        if self._error_occurred:
            return chunk
            
        # Check for error in chunk
        if "data: " in chunk and '"error":' in chunk:
            self._error_occurred = True
            logger.debug("streaming_cache_writer_error_detected", chunk_preview=chunk[:100])
            return chunk
        
        # Cache the chunk
        await self.cache_manager.set_streaming_chunk(self.request_data, chunk)
        self.chunks_written += 1
        
        # If this is the [DONE] marker, finalize the cache
        if chunk.strip() == "data: [DONE]":
            await self.cache_manager.set_streaming_chunk(self.request_data, "", finalize=True)
            logger.info("streaming_cache_writer_finalized", chunks_written=self.chunks_written)
        
        return chunk
    
    async def intercept_stream(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Intercept a stream, cache chunks, and yield them"""
        try:
            async for chunk in stream:
                yield await self.write_and_yield(chunk)
        except Exception as e:
            self._error_occurred = True
            logger.error("streaming_cache_writer_error", error=str(e))
            raise
