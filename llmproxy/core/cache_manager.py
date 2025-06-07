import hashlib
import json
from typing import Optional, Any, Dict
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

    def _should_cache(self, request_data: dict) -> bool:
        """Determine if request should be cached"""
        # Check for cache control directives
        # Support both direct cache parameter and extra_body.cache
        cache_control = request_data.get("cache")
        if not cache_control:
            extra_body = request_data.get("extra_body", {})
            cache_control = extra_body.get("cache")

        # If cache control is specified and no-cache is True, don't cache
        if cache_control and cache_control.get("no-cache", False):
            return False

        return True

    async def get(self, request_data: dict) -> Optional[Any]:
        """Get cached response"""
        if not self._should_cache(request_data):
            return None

        key = self._generate_cache_key(request_data)

        try:
            data = await self.redis.get(key)

            if data:
                self._hits += 1
                logger.info("cache_hit", key=key)
                cached = json.loads(data)
                # Backwards compatibility for old cache format
                if isinstance(cached, dict) and "data" in cached and "stream" in cached:
                    return cached
                return {"stream": False, "data": cached}

            self._misses += 1
            logger.debug("cache_miss", key=key)
            return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e), key=key)
            return None

    async def set(self, request_data: dict, response_data: Any, *, is_streaming: bool = False, metadata: Optional[Dict[str, Any]] = None):
        """Cache response"""
        if not self._should_cache(request_data):
            return

        if not is_streaming and isinstance(response_data, dict) and "error" in response_data:
            return

        key = self._generate_cache_key(request_data)

        payload = {"stream": is_streaming, "data": response_data}
        if metadata:
            payload["meta"] = metadata

        try:
            await self.redis.setex(key, self.ttl, json.dumps(payload))
            logger.debug("cache_set", key=key, ttl=self.ttl)

        except Exception as e:
            logger.error("cache_set_error", error=str(e), key=key)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
        }
