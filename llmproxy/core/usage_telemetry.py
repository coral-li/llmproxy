"""Per-request usage telemetry emitted to a Redis Stream.

The proxy sees every upstream call after load balancing and failover, so it is
the only place that can report which endpoint actually served a request, how
many attempts it took, and whether the proxy cache answered it. Consumers read
the stream with a consumer group and own their own retention.

Records are best effort: a Redis failure here must never fail a proxied
request, so every write is wrapped and logged rather than raised.
"""

import asyncio
import json
import math
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import redis.asyncio as redis

from llmproxy.config_model import UsageStreamParams
from llmproxy.core.logger import get_logger
from llmproxy.core.redis_utils import await_redis_result
from llmproxy.models.endpoint import Endpoint

logger = get_logger(__name__)

#: Ceiling for any single reported token count. Above this the number is not a
#: usage report, and storing it would overflow the consumer's integer columns.
_MAX_TOKEN_COUNT = 2**63 - 1

#: Records buffered before new ones are dropped. Telemetry must never apply
#: back pressure to proxied traffic, so a saturated queue loses records.
_WRITE_QUEUE_SIZE = 10_000

# Chat Completions and Responses report the same quantities under different
# names; normalize to one shape so consumers do not branch per API surface.
_INPUT_TOKEN_KEYS = ("input_tokens", "prompt_tokens")
_OUTPUT_TOKEN_KEYS = ("output_tokens", "completion_tokens")
_INPUT_DETAIL_KEYS = ("input_tokens_details", "prompt_tokens_details")
_OUTPUT_DETAIL_KEYS = ("output_tokens_details", "completion_tokens_details")


def _first_int(source: Mapping[str, Any], keys: tuple) -> int:
    """Return the first key present in ``source`` coerced to a non-negative int.

    Upstream bodies are not trusted to be sane. ``json.loads`` decodes ``1e999``
    to ``inf`` and ``NaN`` to ``nan``, and ``int()`` raises on both, so a single
    odd token count would otherwise escape telemetry and fail a healthy request.
    Absurd magnitudes are clamped rather than stored, because no real usage
    report exceeds a 64-bit count.
    """
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return 0
            return max(min(int(value), _MAX_TOKEN_COUNT), 0)
    return 0


def _first_mapping(source: Mapping[str, Any], keys: tuple) -> Mapping[str, Any]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def normalize_usage(payload: Any) -> Optional[Dict[str, int]]:
    """Extract a normalized token count from any response body carrying usage.

    Accepts the body itself or a Responses API stream event (which nests the
    completed response under ``response``). Returns ``None`` when no usage
    object is present, which is the normal case for intermediate stream chunks.
    """
    if not isinstance(payload, Mapping):
        return None

    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        nested = payload.get("response")
        if isinstance(nested, Mapping):
            usage = nested.get("usage")
    if not isinstance(usage, Mapping):
        return None

    input_details = _first_mapping(usage, _INPUT_DETAIL_KEYS)
    output_details = _first_mapping(usage, _OUTPUT_DETAIL_KEYS)

    input_tokens = _first_int(usage, _INPUT_TOKEN_KEYS)
    output_tokens = _first_int(usage, _OUTPUT_TOKEN_KEYS)
    total_tokens = _first_int(usage, ("total_tokens",)) or (
        input_tokens + output_tokens
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        # Providers report cache reads inside the input detail block; callers
        # price these differently from uncached input.
        "cached_tokens": _first_int(input_details, ("cached_tokens",)),
        "cache_write_tokens": _first_int(
            input_details, ("cache_write_tokens", "cache_creation_tokens")
        ),
        "reasoning_tokens": _first_int(output_details, ("reasoning_tokens",)),
    }


def usage_from_chunks(chunks: List[str]) -> Optional[Dict[str, int]]:
    """Extract the usage report from a replayed SSE stream."""
    observer = StreamUsageObserver()
    for chunk in chunks:
        observer.observe(chunk)
    return observer.usage


def extract_reasoning_effort(request_data: Mapping[str, Any]) -> Optional[str]:
    """Read the requested reasoning effort from either API surface."""
    reasoning = request_data.get("reasoning")
    if isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            return effort

    effort = request_data.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        return effort
    return None


def extract_caller_headers(
    headers: Mapping[str, str], wanted: List[str]
) -> Dict[str, str]:
    """Copy the configured attribution headers, lowercased, from a request.

    Header names are matched case-insensitively because HTTP header casing is
    not significant and clients vary.
    """
    if not headers:
        return {}

    # Normalize once rather than relying on the caller handing us a
    # case-insensitive mapping: a plain dict would otherwise miss every header
    # whose casing does not match the configured name exactly.
    lowered = {
        key.lower(): value for key, value in headers.items() if isinstance(key, str)
    }

    collected: Dict[str, str] = {}
    for name in wanted:
        value = lowered.get(name.lower())
        if isinstance(value, str) and value.strip():
            # Bound the stored value so a hostile or buggy caller cannot push
            # arbitrarily large strings into the stream.
            collected[name.lower()] = value.strip()[:256]
    return collected


@dataclass
class UsageContext:
    """Per-request facts known before the upstream call is made."""

    api_surface: str
    model_group: str
    streaming: bool
    reasoning_effort: Optional[str] = None
    caller: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: Replayed SSE chunks for a streamed cache hit. Held here rather than on
    #: the handler, which is a singleton shared by concurrent requests.
    cached_chunks: Optional[List[str]] = None

    def latency_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)


# Set once per request so handler internals can reach the context without
# threading it through every overridden method. Readers that outlive the
# request (streaming generators) must copy it into a local first, because an
# async generator runs in the context of whoever iterates it.
current_usage_context: ContextVar[Optional[UsageContext]] = ContextVar(
    "llmproxy_usage_context", default=None
)


class StreamUsageObserver:
    """Scan SSE chunks for the terminal usage report without buffering the body.

    Both API surfaces emit usage exactly once, at the end: Responses inside the
    ``response.completed`` event, Chat Completions in a final chunk carrying an
    empty ``choices`` array. Parsing every delta chunk as JSON would be costly
    on long streams, so a substring check gates the parse.
    """

    def __init__(self) -> None:
        self.usage: Optional[Dict[str, int]] = None
        self.endpoint_model: Optional[str] = None

    def observe(self, chunk: str) -> None:
        if not chunk or '"usage"' not in chunk:
            return

        for line in chunk.splitlines():
            if not line.startswith("data:"):
                continue
            # The single space after the colon is optional in the SSE spec.
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue

            usage = normalize_usage(parsed)
            if usage is not None:
                self.usage = usage
                self._capture_model(parsed)

    def _capture_model(self, parsed: Any) -> None:
        if not isinstance(parsed, Mapping):
            return
        model = parsed.get("model")
        if not isinstance(model, str):
            nested = parsed.get("response")
            model = nested.get("model") if isinstance(nested, Mapping) else None
        if isinstance(model, str) and model:
            self.endpoint_model = model


class UsageRecorder:
    """Queue usage records for a background writer.

    ``record`` is deliberately synchronous. It is called from ``finally``
    blocks that can already be inside a cancelled scope — a client that
    disconnects mid-stream is the common case — and any ``await`` there raises
    ``CancelledError`` before the write happens, silently losing the record for
    a stream the upstream has already billed. Handing the record to a queue
    cannot be cancelled and cannot raise.
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        params: Optional[UsageStreamParams],
    ) -> None:
        self.redis = redis_client
        self.params = params
        self._queue: Optional["asyncio.Queue"] = None
        self._task: Optional["asyncio.Task"] = None
        self._dropped = 0

    @property
    def enabled(self) -> bool:
        return (
            self.redis is not None and self.params is not None and self.params.enabled
        )

    def caller_headers(self, headers: Mapping[str, str]) -> Dict[str, str]:
        if not self.enabled or self.params is None:
            return {}
        return extract_caller_headers(headers, self.params.caller_headers)

    def build_context(
        self,
        *,
        api_surface: str,
        model_group: str,
        request_data: Mapping[str, Any],
        caller: Optional[Dict[str, str]] = None,
        start_time: Optional[float] = None,
    ) -> UsageContext:
        return UsageContext(
            api_surface=api_surface,
            model_group=model_group,
            streaming=bool(request_data.get("stream", False)),
            reasoning_effort=extract_reasoning_effort(request_data),
            caller=caller or {},
            start_time=start_time if start_time is not None else time.time(),
        )

    def record(
        self,
        context: UsageContext,
        *,
        status_code: int,
        endpoint: Optional[Endpoint] = None,
        endpoint_model: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        endpoint_base_url: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        cache_hit: bool = False,
        attempts: int = 1,
        error: Optional[str] = None,
    ) -> None:
        """Queue one record for a completed request. Never blocks or raises."""
        if not self.enabled or self.params is None:
            return

        record: Dict[str, Any] = {
            "request_id": context.request_id,
            "recorded_at": time.time(),
            "api_surface": context.api_surface,
            "model_group": context.model_group,
            "endpoint_model": endpoint_model or (endpoint.model if endpoint else None),
            "endpoint_id": endpoint_id or (endpoint.id if endpoint else None),
            "endpoint_base_url": endpoint_base_url
            or (endpoint.base_url if endpoint else None),
            "streaming": context.streaming,
            "reasoning_effort": context.reasoning_effort,
            "status_code": status_code,
            "cache_hit": cache_hit,
            "attempts": attempts,
            "latency_ms": context.latency_ms(),
            "caller": context.caller,
            "usage": usage,
        }
        if error:
            record["error"] = str(error)[:500]

        self._enqueue(record)

    def _enqueue(self, record: Dict[str, Any]) -> None:
        try:
            queue = self._ensure_worker()
        except RuntimeError:
            # No running loop (synchronous teardown); nothing can be written.
            return
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning(
                    "usage_record_queue_full",
                    dropped=self._dropped,
                )

    def _ensure_worker(self) -> "asyncio.Queue":
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=_WRITE_QUEUE_SIZE)
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())
        return self._queue

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            record = await self._queue.get()
            try:
                await self._write(record)
            finally:
                self._queue.task_done()

    async def _write(self, record: Dict[str, Any]) -> None:
        if self.redis is None or self.params is None:
            return
        try:
            await await_redis_result(
                self.redis.xadd(
                    self.params.stream_key,
                    {"payload": json.dumps(record)},
                    maxlen=self.params.max_len,
                    approximate=True,
                )
            )
        except Exception as exc:  # pragma: no cover - telemetry is best effort
            logger.warning(
                "usage_record_write_failed",
                stream_key=self.params.stream_key,
                error=str(exc),
            )

    async def flush(self, timeout: float = 2.0) -> None:
        """Wait for queued records to be written. For tests and shutdown."""
        if self._queue is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("usage_record_flush_timeout")

    async def aclose(self, timeout: float = 2.0) -> None:
        """Drain what is queued, then stop the writer."""
        if self._queue is not None and self._task is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except (asyncio.TimeoutError, Exception):
                pass
        if self._task is not None:
            self._task.cancel()
            self._task = None
