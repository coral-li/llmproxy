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

        try:
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
        reconstructed = []

        # Generate new IDs for this reconstruction
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        for chunk_data in normalized_chunks:
            chunk = EventAwareChunk.from_dict(chunk_data)

            if chunk.event_type == "response.created":
                # Reconstruct response.created event
                event_data = {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created": chunk.metadata.get("created", int(time.time())),
                        "model": chunk.metadata.get("model", ""),
                        "outputs": [],
                    },
                }
                reconstructed.append("event: response.created\n")
                reconstructed.append(f"data: {json.dumps(event_data)}\n")
                reconstructed.append("\n")  # Empty line between events

            elif chunk.event_type == "response.output_item.added":
                # Reconstruct output item event
                event_data = {
                    "type": "response.output_item.added",
                    "output_index": chunk.metadata.get("output_index", 0),
                    "item": {
                        "id": (
                            message_id
                            if chunk.data_type == "message"
                            else f"rs_{uuid.uuid4().hex[:12]}"
                        ),
                        "type": chunk.data_type or "message",
                        "summary": chunk.metadata.get("summary", []),
                    },
                }
                reconstructed.append("event: response.output_item.added\n")
                reconstructed.append(f"data: {json.dumps(event_data)}\n")
                reconstructed.append("\n")

            elif chunk.event_type == "response.output_text.delta":
                # Reconstruct text delta event
                event_data = {
                    "type": "response.output_text.delta",
                    "output_index": chunk.metadata.get("output_index", 0),
                    "content_index": chunk.metadata.get("content_index", 0),
                    "delta": chunk.content or "",
                }
                reconstructed.append("event: response.output_text.delta\n")
                reconstructed.append(f"data: {json.dumps(event_data)}\n")
                reconstructed.append("\n")

            elif chunk.event_type == "response.completed":
                # Reconstruct completed event
                event_data = {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created": chunk.metadata.get("created", int(time.time())),
                        "model": chunk.metadata.get("model", ""),
                        "outputs": chunk.metadata.get("outputs", []),
                    },
                }
                reconstructed.append("event: response.completed\n")
                reconstructed.append(f"data: {json.dumps(event_data)}\n")
                reconstructed.append("\n")

        return reconstructed

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
        if event_type == "response.created":
            return EventAwareChunk(
                event_type=event_type,
                data_type="response",
                metadata={
                    "model": event_data.get("response", {}).get("model"),
                    "created": event_data.get("response", {}).get("created"),
                },
            )

        elif event_type == "response.output_item.added":
            item = event_data.get("item", {})
            return EventAwareChunk(
                event_type=event_type,
                data_type=item.get("type"),
                metadata={
                    "output_index": event_data.get("output_index"),
                    "summary": item.get("summary", []),
                },
            )

        elif event_type == "response.output_text.delta":
            return EventAwareChunk(
                event_type=event_type,
                data_type="text",
                content=event_data.get("delta", ""),
                metadata={
                    "output_index": event_data.get("output_index", 0),
                    "content_index": event_data.get("content_index", 0),
                },
            )

        elif event_type == "response.completed":
            response = event_data.get("response", {})
            outputs = []
            for output in response.get("outputs", []):
                # Filter out dynamic IDs
                cleaned_output = {
                    "type": output.get("type"),
                    "content": output.get("content", []),
                }
                outputs.append(cleaned_output)

            return EventAwareChunk(
                event_type=event_type,
                data_type="response",
                metadata={
                    "model": response.get("model"),
                    "created": response.get("created"),
                    "outputs": outputs,
                },
            )

        return None

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

        # Don't cache if error occurred
        if self._error_occurred:
            return chunk

        # Check for error in chunk - be more specific to avoid false positives
        if self._detect_error_in_chunk(chunk):
            return chunk

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
        key = f"{self.cache_manager._generate_cache_key(self.request_data)}:stream"
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
            logger.error("chat_streaming_cache_error", error=str(e))

    async def _finalize_responses_cache(self) -> None:
        """Finalize the responses API cache with normalized chunks."""
        key = f"{self.cache_manager._generate_cache_key(self.request_data)}:responses_stream"

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
                yield await self.write_and_yield(chunk)
        except Exception as e:
            self._error_occurred = True
            logger.error("streaming_cache_writer_error", error=str(e))
            raise
