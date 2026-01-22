import hashlib
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import redis.asyncio as redis

from .logger import get_logger

logger = get_logger(__name__)


class EventAwareChunk:
    """Represents a normalized chunk for caching that abstracts API differences"""

    def __init__(
        self,
        event_type: Optional[str] = None,
        data_type: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.event_type = event_type  # For responses API: response.created, response.output_text.delta, etc.
        self.data_type = data_type  # For responses API: message, reasoning, etc.
        self.content = content  # The actual content text
        self.metadata = metadata or {}  # Additional fields that should be preserved

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "event_type": self.event_type,
            "data_type": self.data_type,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventAwareChunk":
        """Create from dictionary"""
        return cls(
            event_type=data.get("event_type"),
            data_type=data.get("data_type"),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
        )


class _ResponsesStreamRebuilder:
    """Helper to rebuild responses API SSE sequences from normalized chunks."""

    def __init__(self) -> None:
        self.response_id: Optional[str] = None
        self.output_item_ids: Dict[int, str] = {}
        self.chunks: List[str] = []

    def _ensure_response_id(self, metadata: dict) -> str:
        candidate: Optional[str] = None
        if isinstance(metadata, dict):
            candidate = metadata.get("response_id") or metadata.get("id")

        if isinstance(candidate, str) and candidate:
            self.response_id = candidate

        if self.response_id is None:
            self.response_id = f"resp_{uuid.uuid4().hex[:12]}"

        return self.response_id

    def _append_event(self, name: str, payload: Dict[str, Any]) -> None:
        self.chunks.append(f"event: {name}\n")
        self.chunks.append(f"data: {json.dumps(payload)}\n")
        self.chunks.append("\n")

    def handle_created(self, metadata: dict, _: EventAwareChunk) -> None:
        current_response_id = self._ensure_response_id(metadata)
        created_ts = metadata.get("created_at")
        if created_ts is None:
            created_ts = metadata.get("created", int(time.time()))
        created_value = (
            created_ts if isinstance(created_ts, (int, float)) else int(time.time())
        )

        response_payload: Dict[str, Any] = {
            "id": current_response_id,
            "object": "response",
            "created_at": int(created_value),
            "model": metadata.get("model", ""),
            "outputs": [],
        }

        if "created" in metadata:
            response_payload["created"] = metadata.get("created")

        if "status" in metadata:
            response_payload["status"] = metadata.get("status")

        if "status_details" in metadata:
            response_payload["status_details"] = metadata.get("status_details")

        event_payload = {
            "type": "response.created",
            "response": response_payload,
        }

        self._append_event("response.created", event_payload)

    def handle_output_item_added(self, metadata: dict, chunk: EventAwareChunk) -> None:
        current_response_id = self._ensure_response_id(metadata)
        output_index = metadata.get("output_index", 0)

        stored_item_id = metadata.get("item_id")
        if not isinstance(stored_item_id, str) or not stored_item_id:
            stored_item_id = self.output_item_ids.get(output_index)

        if not stored_item_id:
            prefix = "msg" if chunk.data_type == "message" else "rs"
            stored_item_id = f"{prefix}_{uuid.uuid4().hex[:12]}"

        self.output_item_ids[output_index] = stored_item_id

        item_payload: Dict[str, Any] = {
            "id": stored_item_id,
            "type": chunk.data_type or "message",
            "summary": metadata.get("summary", []),
        }

        if "item_status" in metadata:
            item_payload["status"] = metadata.get("item_status")

        if "item_role" in metadata:
            item_payload["role"] = metadata.get("item_role")

        if "item_name" in metadata:
            item_payload["name"] = metadata.get("item_name")

        event_payload: Dict[str, Any] = {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": item_payload,
        }

        if current_response_id:
            event_payload["response_id"] = current_response_id

        self._append_event("response.output_item.added", event_payload)

    def handle_output_text_delta(self, metadata: dict, chunk: EventAwareChunk) -> None:
        event_payload: Dict[str, Any] = {
            "type": "response.output_text.delta",
            "output_index": metadata.get("output_index", 0),
            "content_index": metadata.get("content_index", 0),
            "delta": chunk.content or "",
        }

        response_id = metadata.get("response_id")
        if isinstance(response_id, str) and response_id:
            self._ensure_response_id(metadata)
            event_payload["response_id"] = response_id

        self._append_event("response.output_text.delta", event_payload)

    def handle_completed(self, metadata: dict, _: EventAwareChunk) -> None:
        current_response_id = self._ensure_response_id(metadata)
        created_ts = metadata.get("created_at")
        if created_ts is None:
            created_ts = metadata.get("created", int(time.time()))
        created_value = (
            created_ts if isinstance(created_ts, (int, float)) else int(time.time())
        )

        outputs: List[Dict[str, Any]] = []
        stored_outputs = metadata.get("outputs", [])
        if isinstance(stored_outputs, list):
            for idx, output in enumerate(stored_outputs):
                if not isinstance(output, dict):
                    continue

                output_copy: Dict[str, Any] = dict(output)
                output_id = output_copy.get("id")
                fallback_id = self.output_item_ids.get(idx)

                if (not isinstance(output_id, str) or not output_id) and fallback_id:
                    output_copy["id"] = fallback_id

                outputs.append(output_copy)

        response_payload: Dict[str, Any] = {
            "id": current_response_id,
            "object": "response",
            "created_at": int(created_value),
            "model": metadata.get("model", ""),
            "outputs": outputs,
        }

        if "created" in metadata:
            response_payload["created"] = metadata.get("created")

        if "status" in metadata:
            response_payload["status"] = metadata.get("status")

        if "status_details" in metadata:
            response_payload["status_details"] = metadata.get("status_details")

        event_payload = {
            "type": "response.completed",
            "response": response_payload,
        }

        self._append_event("response.completed", event_payload)


class StreamingCacheAborted(RuntimeError):
    """Raised when streaming cache must halt due to an upstream error."""

    def __init__(self, chunk: Optional[str], error: Optional[dict] = None):
        super().__init__("Streaming cache aborted after upstream error chunk")
        self.chunk = chunk
        self.error = error or {}


class CacheManager:
    """Manages LLM response caching with intelligent key generation"""

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl: int = 604800,
        namespace: str = "llmproxy",
        cache_enabled: bool = True,
    ):
        self.redis = redis_client
        self.ttl = ttl
        self.namespace = namespace
        self.cache_enabled = cache_enabled  # Global cache setting from config
        self._hits = 0
        self._misses = 0
        self._streaming_hits = 0
        self._streaming_misses = 0

    def _generate_cache_key(self, request_data: dict) -> str:
        """Generate cache key from request parameters"""
        # Create deterministic hash
        cache_str = json.dumps(request_data, sort_keys=True)
        cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()

        key = f"{self.namespace}:{cache_hash}"

        return key

    def _should_cache(self, request_data: dict) -> bool:
        """Determine if request should be cached"""
        # Check for cache control directives
        # Support both direct cache parameter and extra_body.cache
        cache_control = request_data.get("cache")
        if not cache_control:
            # Check extra_body for cache control
            extra_body = request_data.get("extra_body", {})
            if not isinstance(extra_body, dict):
                extra_body = {}
            cache_control = extra_body.get("cache")

        logger.debug(
            "should_cache_check",
            cache_control=cache_control,
            is_streaming=request_data.get("stream", False),
            has_extra_body="extra_body" in request_data,
            global_cache_enabled=self.cache_enabled,
        )

        # -------------------------------------------------------------
        # Per-request override: only support dict with "no-cache": bool
        # -------------------------------------------------------------

        if isinstance(cache_control, dict):
            if cache_control.get("no-cache", False):
                logger.debug(
                    "should_cache_result", result=False, reason="no-cache directive"
                )
                return False

            # Any dict without "no-cache" simply defers to global setting.
            logger.debug(
                "should_cache_result",
                result=self.cache_enabled,
                reason="cache control dict without no-cache",
            )
            return self.cache_enabled

        # For all other cases (including booleans), ignore directive and use global.
        logger.debug(
            "should_cache_result",
            result=self.cache_enabled,
            reason="no per-request cache directive or unsupported type",
        )
        return self.cache_enabled

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
                return json.loads(data)  # type: ignore

            self._misses += 1
            logger.debug("cache_miss", key=key)
            return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e), key=key)
            return None

    def _is_empty_response(self, response_data: dict) -> bool:
        """Return True if the response from the LLM is considered *empty* and therefore should not be cached.

        For chat completions style responses (OpenAI compatible) a response is treated as
        empty when *all* choices are empty – i.e. ``message.content``, ``delta.content``, or ``text`` is
        either ``None`` or the empty string after stripping.  A missing ``choices`` key
        is also interpreted as an empty response.

        For responses API format, checks the ``outputs`` structure for meaningful content.

        For embeddings API format, checks if the response has valid embeddings data.
        """

        # Check for responses API format first
        # Accept both 'outputs' (preferred) and 'output' (alternate), or direct 'output_text'
        if (
            "outputs" in response_data
            or "output" in response_data
            or (isinstance(response_data.get("output_text"), str))
        ):
            return self._is_empty_responses_api(response_data)

        # Check for embeddings API format (has 'data' with embedding objects)
        if self._is_embeddings_response(response_data):
            return self._is_empty_embeddings(response_data)

        # Handle chat completions format (has 'choices')
        return self._is_empty_chat_completions(response_data)

    def _is_embeddings_response(self, response_data: dict) -> bool:
        """Check if this is an embeddings API response format."""
        return (
            response_data.get("object") == "list"
            and "data" in response_data
            and isinstance(response_data.get("data"), list)
            and len(response_data.get("data", [])) > 0
            and response_data.get("data", [{}])[0].get("object") == "embedding"
        )

    def _is_empty_embeddings(self, response_data: dict) -> bool:
        """Check if embeddings response is empty."""
        data = response_data.get("data", [])
        if not isinstance(data, list) or len(data) == 0:
            logger.error(
                "empty_embeddings_response data is not a list",
                response_data=response_data,
            )
            return True

        # Check if any embedding data exists
        for item in data:
            if isinstance(item, dict):
                embedding = item.get("embedding")
                if embedding is not None:
                    return False

        logger.error(
            "empty_embeddings_response no meaningful embedding data found",
            response_data=response_data,
        )
        # No meaningful embedding data found
        return True

    def _is_empty_responses_api(self, response_data: dict) -> bool:
        """Check if responses API format response is empty.

        Supports both "outputs" (preferred) and "output" (singular) response shapes
        observed across providers/mocks, and also a direct "output_text" shorthand
        when present.
        """
        # First, support the convenience top-level output_text field
        output_text = response_data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return False

        # Normalize outputs list from either key
        outputs = response_data.get("outputs")
        if outputs is None:
            outputs = response_data.get("output", [])

        if not isinstance(outputs, list) or len(outputs) == 0:
            return True

        for output in outputs:
            if not isinstance(output, dict):
                continue

            if self._has_meaningful_output_content(output):
                return False

        # If we reach here, no output had meaningful content
        return True

    def _has_meaningful_output_content(self, output: dict) -> bool:
        """Check if a single output item has meaningful content."""
        output_type = output.get("type")

        if output_type == "message":
            # Message output format: outputs[].content[].text
            content_items = output.get("content", [])
            if isinstance(content_items, list):
                for content_item in content_items:
                    if isinstance(content_item, dict):
                        text_content = content_item.get("text")
                        if isinstance(text_content, str) and text_content.strip():
                            return True

        elif output_type in ["text", "output_text"]:
            # Direct text output
            text_content = output.get("text")
            if isinstance(text_content, str) and text_content.strip():
                return True

        # Additional check for any other potential text fields
        for field in ["content", "text", "data"]:
            field_value = output.get(field)
            if isinstance(field_value, str) and field_value.strip():
                return True

        if "type" in output and output["type"] == "function_call":
            return True

        return False

    def _is_empty_chat_completions(self, response_data: dict) -> bool:
        """Check if chat completions format response is empty."""
        if "choices" not in response_data:
            return True

        choices = response_data.get("choices", [])
        if not isinstance(choices, list) or len(choices) == 0:
            return True

        for choice in choices:
            if self._has_meaningful_choice_content(choice):
                return False

        # Reaching here means no choice had meaningful content
        return True

    def _has_meaningful_choice_content(self, choice: Any) -> bool:
        """Check if a single choice has meaningful content."""
        if not isinstance(choice, dict):
            return False

        content: Optional[str] = None

        # Chat completion format { "message": {"content": "..."} }
        message = (
            choice.get("message") if isinstance(choice.get("message"), dict) else None
        )
        if message is not None:
            content = message.get("content")

            # Check for tool calls in message (modern format)
            tool_calls = message.get("tool_calls")
            if self._has_meaningful_tool_calls(tool_calls):
                logger.debug("meaningful_content_detected", reason="message.tool_calls")
                return True

            # Check for function call in message (legacy format)
            function_call = message.get("function_call")
            if self._has_meaningful_function_call(function_call):
                logger.debug(
                    "meaningful_content_detected", reason="message.function_call"
                )
                return True

        # Streaming chat completion format { "delta": {"content": "..."} }
        if content is None:
            delta = (
                choice.get("delta") if isinstance(choice.get("delta"), dict) else None
            )
            if delta is not None:
                content = delta.get("content")

                # Check for tool calls in delta (streaming format)
                tool_calls = delta.get("tool_calls")
                if self._has_meaningful_tool_calls(tool_calls):
                    logger.debug(
                        "meaningful_content_detected", reason="delta.tool_calls"
                    )
                    return True

                # Check for function call in delta (streaming legacy format)
                function_call = delta.get("function_call")
                if self._has_meaningful_function_call(function_call):
                    logger.debug(
                        "meaningful_content_detected", reason="delta.function_call"
                    )
                    return True

        # Legacy / completions format uses "text"
        if content is None:
            content = choice.get("text")

        # If this choice contains non-empty content we deem it meaningful
        if isinstance(content, str) and bool(content.strip()):
            logger.debug("meaningful_content_detected", reason="textual_content")
            return True

        return False

    def _has_meaningful_tool_calls(self, tool_calls: Any) -> bool:
        """Check if tool_calls contains meaningful data."""
        if not isinstance(tool_calls, list) or len(tool_calls) == 0:
            return False

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            # Check if tool call has meaningful function data
            function = tool_call.get("function")
            if isinstance(function, dict):
                # Must have a function name
                if function.get("name"):
                    return True

            # Check if tool call has other meaningful data (type, id, etc.)
            if tool_call.get("type") or tool_call.get("id"):
                return True

        return False

    def _has_meaningful_function_call(self, function_call: Any) -> bool:
        """Check if function_call contains meaningful data (legacy format)."""
        if not isinstance(function_call, dict):
            return False

        # Must have a function name to be meaningful
        return bool(function_call.get("name"))

    async def set(self, request_data: dict, response_data: dict) -> None:
        """Cache response for non-streaming requests"""
        if not self._should_cache(request_data):
            return

        # Skip caching empty model responses so callers can retry later
        if self._is_empty_response(response_data):
            logger.info(
                "cache_skip_empty_response",
                request_summary={"model": request_data.get("model")},
                response_data=response_data,
            )
            return

        key = self._generate_cache_key(request_data)

        try:
            await self.redis.setex(key, self.ttl, json.dumps(response_data))
            logger.debug("cache_set", key=key, ttl=self.ttl)

        except Exception as e:
            logger.error("cache_set_error", error=str(e), key=key)

    async def get_streaming(self, request_data: dict) -> Optional[List[str]]:
        """Get cached streaming response chunks"""
        if not self._should_cache(request_data):
            return None

        # Determine if this is a responses API request
        is_responses_api = "input" in request_data and "messages" not in request_data
        key_suffix = ":responses_stream" if is_responses_api else ":stream"
        key = f"{self._generate_cache_key(request_data)}{key_suffix}"
        sentinel_key = f"{key}:error"

        try:
            sentinel_value = await self.redis.get(sentinel_key)
            if sentinel_value:
                self._streaming_misses += 1
                logger.debug(
                    "streaming_cache_bypassed_due_to_error",
                    key=key,
                    api_type="responses" if is_responses_api else "chat",
                )
                return None

            # For responses API, get normalized chunks
            if is_responses_api:
                normalized_data = await self.redis.get(f"{key}:normalized")
                if normalized_data:
                    normalized_chunks = json.loads(normalized_data)
                    # Reconstruct SSE stream from normalized chunks
                    reconstructed_chunks = self._reconstruct_responses_stream(
                        normalized_chunks
                    )

                    self._streaming_hits += 1
                    logger.info(
                        "streaming_cache_hit",
                        key=key,
                        api_type="responses",
                        num_normalized_chunks=len(normalized_chunks),
                        num_reconstructed_chunks=len(reconstructed_chunks),
                    )
                    return reconstructed_chunks
            else:
                # For chat completions, use the existing raw chunk approach
                chunks = await self.redis.lrange(key, 0, -1)  # type: ignore
                if chunks:
                    self._streaming_hits += 1
                    logger.info(
                        "streaming_cache_hit",
                        key=key,
                        api_type="chat",
                        num_chunks=len(chunks),
                    )
                    return list(chunks)

            self._streaming_misses += 1
            logger.debug(
                "streaming_cache_miss",
                key=key,
                api_type="responses" if is_responses_api else "chat",
            )
            return None

        except Exception as e:
            logger.error("streaming_cache_get_error", error=str(e), key=key)
            return None

    def _reconstruct_responses_stream(self, normalized_chunks: List[dict]) -> List[str]:
        """Reconstruct responses API SSE stream from normalized chunks"""
        rebuilder = _ResponsesStreamRebuilder()

        handlers = {
            "response.created": rebuilder.handle_created,
            "response.output_item.added": rebuilder.handle_output_item_added,
            "response.output_text.delta": rebuilder.handle_output_text_delta,
            "response.completed": rebuilder.handle_completed,
        }

        for chunk_data in normalized_chunks:
            chunk = EventAwareChunk.from_dict(chunk_data)
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            event_type = chunk.event_type
            if not isinstance(event_type, str):
                continue
            handler = handlers.get(event_type)
            if handler:
                handler(metadata, chunk)

        return rebuilder.chunks

    async def set_streaming_chunk(
        self, request_data: dict, chunk: str, finalize: bool = False
    ) -> None:
        """Cache a streaming response chunk"""
        if not self._should_cache(request_data):
            return

        # This method is now handled by StreamingCacheWriter
        logger.debug("set_streaming_chunk_deprecated", chunk_preview=chunk[:50])

    async def create_streaming_cache_writer(
        self, request_data: dict
    ) -> "StreamingCacheWriter":
        """Create a streaming cache writer that intercepts and caches chunks"""
        # Determine if this is a responses API request
        is_responses_api = "input" in request_data and "messages" not in request_data
        return StreamingCacheWriter(
            self, request_data, is_responses_api=is_responses_api
        )

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        streaming_total = self._streaming_hits + self._streaming_misses
        streaming_hit_rate = (
            (self._streaming_hits / streaming_total * 100) if streaming_total > 0 else 0
        )

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
        """Invalidate all cached entries for this namespace using SCAN + batched UNLINK/DEL"""
        try:
            pattern = f"{self.namespace}:*"
            batch_size = 500

            total_deleted = 0
            batch: List[str] = []

            async for key in self.redis.scan_iter(match=pattern, count=batch_size):
                batch.append(key)
                if len(batch) >= batch_size:
                    try:
                        if hasattr(self.redis, "unlink"):
                            deleted = await self.redis.unlink(*batch)
                        else:
                            deleted = await self.redis.delete(*batch)
                        total_deleted += int(deleted)
                    finally:
                        batch = []

            # Flush any remaining keys
            if batch:
                try:
                    if hasattr(self.redis, "unlink"):
                        deleted = await self.redis.unlink(*batch)
                    else:
                        deleted = await self.redis.delete(*batch)
                    total_deleted += int(deleted)
                finally:
                    batch = []

            logger.info(
                "cache_invalidate_all",
                namespace=self.namespace,
                keys_deleted=total_deleted,
                method="scan_iter",
            )
            return total_deleted

        except Exception as e:
            logger.error(
                "cache_invalidate_all_error", error=str(e), namespace=self.namespace
            )
            return 0

    # Removed invalidate_request as it was unused in production code and led to duplicated logic


class StreamingCacheWriter:
    """Helper class to intercept and cache streaming chunks"""

    def __init__(
        self,
        cache_manager: CacheManager,
        request_data: dict,
        is_responses_api: bool = False,
    ):
        self.cache_manager = cache_manager
        self.request_data = request_data
        self.chunks_written = 0
        self._error_occurred = False
        self.is_responses_api = is_responses_api
        self.normalized_chunks: Optional[List[dict]] = None
        self.current_event = None
        self.buffered_lines: List[str] = []
        # Indicates if we have observed any non-empty content so that we only cache
        # meaningful responses.
        self._has_content: bool = False
        self._error_payload: Optional[dict] = None
        self._sentinel_cleared = False

    def _stream_key(self) -> str:
        suffix = ":responses_stream" if self.is_responses_api else ":stream"
        return f"{self.cache_manager._generate_cache_key(self.request_data)}{suffix}"

    def _error_sentinel_key(self) -> str:
        return f"{self._stream_key()}:error"

    async def _clear_failure_sentinel(self) -> None:
        if self._sentinel_cleared:
            return

        try:
            await self.cache_manager.redis.delete(self._error_sentinel_key())
            logger.debug(
                "streaming_cache_sentinel_cleared",
                key=self._error_sentinel_key(),
            )
        except Exception as sentinel_error:  # pragma: no cover - best effort cleanup
            logger.debug(
                "streaming_cache_sentinel_clear_failed",
                key=self._error_sentinel_key(),
                error=str(sentinel_error),
            )
        finally:
            self._sentinel_cleared = True

    async def _mark_stream_failure(self) -> None:
        try:
            await self.cache_manager.redis.setex(
                self._error_sentinel_key(),
                self.cache_manager.ttl,
                "1",
            )
            logger.debug(
                "streaming_cache_failure_sentinel_set",
                key=self._error_sentinel_key(),
                error_payload=self._error_payload,
            )
        except Exception as sentinel_error:  # pragma: no cover - best effort cleanup
            logger.debug(
                "streaming_cache_failure_sentinel_error",
                key=self._error_sentinel_key(),
                error=str(sentinel_error),
            )

    async def _handle_stream_cleanup(self) -> None:
        """Remove any partially written cache artifacts when a stream fails."""
        cleanup_keys = [self._stream_key()]
        if self.is_responses_api:
            cleanup_keys.append(f"{cleanup_keys[0]}:normalized")

        try:
            await self.cache_manager.redis.delete(*cleanup_keys)
            logger.debug(
                "streaming_cache_cleanup",
                keys=cleanup_keys,
            )
        except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
            logger.debug(
                "streaming_cache_cleanup_failed",
                keys=cleanup_keys,
                error=str(cleanup_error),
            )
        finally:
            await self._mark_stream_failure()
            self.buffered_lines = []
            self.normalized_chunks = None
            self._has_content = False

    def _parse_sse_lines(self, chunk: str) -> Tuple[Optional[str], Optional[dict]]:
        """Parse SSE lines to extract event type and data"""
        lines = chunk.strip().split("\n")
        event_type = None
        data_lines = []

        # First pass: collect all event and data lines
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_lines.append(line[6:])

        # Second pass: try to parse the first valid JSON data line
        event_data = None
        for data_content in data_lines:
            try:
                event_data = json.loads(data_content.strip())
                break  # Use the first successfully parsed data line
            except json.JSONDecodeError:
                logger.debug(
                    "parse_responses_event_json_error", data_content=data_content
                )
                continue
        return event_type, event_data

    def _create_response_chunk(
        self, event_type: str, event_data: dict
    ) -> Optional[EventAwareChunk]:
        """Create EventAwareChunk based on event type and data"""
        handlers = {
            "response.created": self._normalize_response_created,
            "response.output_item.added": self._normalize_output_item_added,
            "response.output_text.delta": self._normalize_output_text_delta,
            "response.completed": self._normalize_response_completed,
        }

        handler = handlers.get(event_type)
        if handler:
            return handler(event_data)
        return None

    def _extract_created_metadata(
        self, response_payload: dict
    ) -> Tuple[Optional[Any], Optional[Any]]:
        created_at = response_payload.get("created_at")
        created = response_payload.get("created")
        if created_at is None:
            created_at = created
        return created_at, created

    def _normalize_response_created(
        self, event_data: dict
    ) -> Optional[EventAwareChunk]:
        response_payload = (
            event_data.get("response", {}) if isinstance(event_data, dict) else {}
        )
        created_at, created = self._extract_created_metadata(response_payload)

        metadata = {
            "model": response_payload.get("model"),
            "created_at": created_at,
        }

        if created is not None:
            metadata["created"] = created

        response_id = response_payload.get("id")
        if response_id:
            metadata["response_id"] = response_id

        if "status" in response_payload:
            metadata["status"] = response_payload.get("status")

        if "status_details" in response_payload:
            metadata["status_details"] = response_payload.get("status_details")

        return EventAwareChunk(
            event_type="response.created",
            data_type="response",
            metadata=metadata,
        )

    def _normalize_output_item_added(
        self, event_data: dict
    ) -> Optional[EventAwareChunk]:
        item = event_data.get("item", {}) if isinstance(event_data, dict) else {}

        response_id = event_data.get("response_id")
        if not response_id:
            response_info = event_data.get("response")
            if isinstance(response_info, dict):
                response_id = response_info.get("id")

        metadata = {
            "output_index": event_data.get("output_index"),
            "summary": item.get("summary", []),
        }

        item_id = item.get("id")
        if item_id:
            metadata["item_id"] = item_id

        if response_id:
            metadata["response_id"] = response_id

        if "status" in item:
            metadata["item_status"] = item.get("status")

        if "role" in item:
            metadata["item_role"] = item.get("role")

        if "name" in item:
            metadata["item_name"] = item.get("name")

        return EventAwareChunk(
            event_type="response.output_item.added",
            data_type=item.get("type"),
            metadata=metadata,
        )

    def _normalize_output_text_delta(
        self, event_data: dict
    ) -> Optional[EventAwareChunk]:
        metadata = {
            "output_index": event_data.get("output_index", 0),
            "content_index": event_data.get("content_index", 0),
        }

        if event_data.get("response_id"):
            metadata["response_id"] = event_data.get("response_id")

        return EventAwareChunk(
            event_type="response.output_text.delta",
            data_type="text",
            content=event_data.get("delta", ""),
            metadata=metadata,
        )

    def _normalize_response_completed(
        self, event_data: dict
    ) -> Optional[EventAwareChunk]:
        response = (
            event_data.get("response", {}) if isinstance(event_data, dict) else {}
        )
        created_at, created = self._extract_created_metadata(response)
        outputs = self._clean_completed_outputs(response.get("outputs", []))

        metadata = {
            "model": response.get("model"),
            "created_at": created_at,
            "outputs": outputs,
        }

        if created is not None:
            metadata["created"] = created

        if response.get("id"):
            metadata["response_id"] = response.get("id")

        if "status" in response:
            metadata["status"] = response.get("status")

        if "status_details" in response:
            metadata["status_details"] = response.get("status_details")

        return EventAwareChunk(
            event_type="response.completed",
            data_type="response",
            metadata=metadata,
        )

    def _clean_completed_outputs(self, outputs: Any) -> List[dict]:
        cleaned_outputs: List[dict] = []
        if not isinstance(outputs, list):
            return cleaned_outputs

        for output in outputs:
            if not isinstance(output, dict):
                continue

            cleaned_output = {
                "type": output.get("type"),
                "content": output.get("content", []),
            }

            if output.get("id"):
                cleaned_output["id"] = output.get("id")

            if "status" in output:
                cleaned_output["status"] = output.get("status")

            if "role" in output:
                cleaned_output["role"] = output.get("role")

            if "summary" in output:
                cleaned_output["summary"] = output.get("summary")

            if "metadata" in output:
                cleaned_output["metadata"] = output.get("metadata")

            cleaned_outputs.append(cleaned_output)

        return cleaned_outputs

    def _parse_responses_event(self, chunk: str) -> Optional[EventAwareChunk]:
        """Parse a responses API SSE event and return normalized chunk"""
        event_type, event_data = self._parse_sse_lines(chunk)

        if not event_type or not event_data:
            return None

        return self._create_response_chunk(event_type, event_data)

    async def write_and_yield(self, chunk: str) -> str:
        """Write chunk to cache and yield it"""
        logger.debug(
            "streaming_cache_writer_chunk_received",
            is_responses_api=self.is_responses_api,
            chunk_preview=chunk[:100] if chunk else None,
            chunk_type=type(chunk).__name__,
            error_occurred=self._error_occurred,
        )

        await self._clear_failure_sentinel()

        # Don't cache if error occurred
        if self._error_occurred:
            raise StreamingCacheAborted(chunk, self._error_payload)

        # Check for error in chunk - be more specific to avoid false positives
        if self._detect_error_in_chunk(chunk):
            await self._handle_stream_cleanup()
            raise StreamingCacheAborted(chunk, self._error_payload)

        if self.is_responses_api:
            await self._handle_responses_api(chunk)
        else:
            await self._handle_chat_completions(chunk)

        return chunk

    def _detect_error_in_chunk(self, chunk: str) -> bool:
        if "data: " in chunk:
            data_start = chunk.find("data: ")
            if data_start != -1:
                data_part = chunk[data_start + 6 :]
                try:
                    data_json = json.loads(data_part.strip())
                    if isinstance(data_json, dict) and "error" in data_json:
                        self._error_occurred = True
                        self._error_payload = data_json
                        logger.debug(
                            "streaming_cache_writer_error_detected",
                            error=data_json.get("error"),
                        )
                        return True
                except json.JSONDecodeError:
                    pass
        return False

    async def _handle_responses_api(self, chunk: str) -> None:
        self.buffered_lines.append(chunk)
        logger.debug(
            "responses_api_buffering_chunk",
            buffer_size=len(self.buffered_lines),
            is_empty_line=chunk.strip() == "",
            is_completion=chunk.strip() == "event: response.completed",
        )
        # SSE events are separated by a blank line. Previously we also flushed
        # when the `event: response.completed` line was seen, which caused the
        # final event to be processed before its data line arrived. This
        # resulted in missing completion metadata in the cache.  We now flush
        # only when we see the blank line that terminates the event.
        if chunk.strip() == "":
            full_event = "\n".join(self.buffered_lines)
            logger.debug(
                "responses_api_processing_event",
                event_preview=full_event[:200],
                buffer_lines=len(self.buffered_lines),
            )
            normalized_chunk = self._parse_responses_event(full_event)
            if normalized_chunk:
                if self.normalized_chunks is None:
                    self.normalized_chunks = []
                self.normalized_chunks.append(normalized_chunk.to_dict())
                self.chunks_written += 1
                # Mark as having content if this chunk contains any text delta
                if normalized_chunk.content is not None:
                    if (
                        isinstance(normalized_chunk.content, str)
                        and normalized_chunk.content.strip()
                    ):
                        self._has_content = True
                logger.debug(
                    "responses_api_chunk_normalized",
                    event_type=normalized_chunk.event_type,
                    has_content=bool(normalized_chunk.content),
                    chunk_index=self.chunks_written,
                )
            self.buffered_lines = []
            if "event: response.completed" in full_event:
                await self._finalize_responses_cache()

    async def _handle_chat_completions(self, chunk: str) -> None:
        key = self._stream_key()
        try:
            # Detect if chunk carries non-empty content; this is a best-effort
            # heuristic based on the OpenAI chat completions streaming format.
            if chunk.startswith("data: "):
                json_part = chunk[6:].strip()
                if json_part and json_part != "[DONE]":
                    try:
                        payload = json.loads(json_part)
                        if isinstance(payload, dict):
                            for choice in payload.get("choices", []):
                                delta = (
                                    choice.get("delta", {})
                                    if isinstance(choice, dict)
                                    else {}
                                )
                                content = (
                                    delta.get("content")
                                    if isinstance(delta, dict)
                                    else None
                                )
                                if isinstance(content, str) and content.strip():
                                    self._has_content = True
                                    break
                    except json.JSONDecodeError:
                        pass

            await self.cache_manager.redis.rpush(key, chunk)  # type: ignore
            # Ensure the list does not linger forever if the stream aborts before [DONE].
            await self.cache_manager.redis.expire(key, self.cache_manager.ttl)
            self.chunks_written += 1
            if chunk.strip() == "data: [DONE]":
                if self._has_content:
                    await self.cache_manager.redis.expire(key, self.cache_manager.ttl)
                    logger.info(
                        "chat_streaming_cache_finalized",
                        chunks_written=self.chunks_written,
                        key=key,
                    )
                else:
                    # No meaningful content – remove the list so that future calls can retry.
                    await self.cache_manager.redis.delete(key)
                    logger.info("chat_streaming_cache_skipped_empty", key=key)
        except Exception as e:
            # Redis failures should not crash the stream; they'll be logged upstream.
            logger.error("chat_streaming_cache_error", error=str(e))
            await self._handle_stream_cleanup()

    async def _finalize_responses_cache(self) -> None:
        """Finalize the responses API cache with normalized chunks."""
        key = self._stream_key()

        try:
            if not self._has_content and not self._infer_content_from_completed_event():
                logger.info("responses_streaming_cache_skipped_empty", key=key)
                return

            normalized_key = f"{key}:normalized"
            await self.cache_manager.redis.setex(
                normalized_key,
                self.cache_manager.ttl,
                json.dumps(self.normalized_chunks),
            )

            logger.info(
                "responses_streaming_cache_finalized",
                chunks_written=self.chunks_written,
                normalized_chunks=(
                    len(self.normalized_chunks) if self.normalized_chunks else 0
                ),
                key=key,
            )
        except Exception as e:
            logger.error(
                "responses_streaming_cache_finalize_error", error=str(e), key=key
            )
            return None
            pass

    def _infer_content_from_completed_event(self) -> bool:
        """Infer presence of meaningful content from a response.completed event."""
        if not isinstance(self.normalized_chunks, list):
            return False

        chunks_list: List[dict] = [
            ch for ch in self.normalized_chunks if isinstance(ch, dict)
        ]
        for chunk in chunks_list:
            if chunk.get("event_type") != "response.completed":
                continue
            metadata = chunk.get("metadata", {})
            outputs = metadata.get("outputs", [])
            outputs_list: List[dict] = [o for o in outputs if isinstance(o, dict)]
            for output in outputs_list:
                content_items = output.get("content", [])
                items_list: List[dict] = [
                    p for p in content_items if isinstance(p, dict)
                ]
                for piece in items_list:
                    text_val = piece.get("text")
                    if isinstance(text_val, str) and text_val.strip():
                        return True
        return False

    async def intercept_stream(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Intercept a stream, cache chunks, and yield them"""
        try:
            async for chunk in stream:
                try:
                    processed_chunk = await self.write_and_yield(chunk)
                except StreamingCacheAborted as aborted:
                    if aborted.chunk is not None:
                        yield aborted.chunk
                    logger.debug(
                        "streaming_cache_aborted",
                        error_payload=aborted.error,
                    )
                    return
                else:
                    yield processed_chunk
        except Exception as e:
            self._error_occurred = True
            await self._handle_stream_cleanup()
            logger.error("streaming_cache_writer_error", error=str(e))
            raise
