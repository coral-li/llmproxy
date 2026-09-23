"""Per-request usage telemetry emitted to a Redis Stream.

The proxy sees every upstream call after load balancing and failover, so it is
the only place that can report which endpoint actually served a request, how
many attempts it took, and whether the proxy cache answered it. Consumers read
the stream from a position they track themselves and own their own retention;
see docs/usage-telemetry.md.

Records are best effort: a Redis failure here must never fail a proxied
request, so every write is wrapped and logged rather than raised.

Records carry counts and labels only. Upstream error bodies can quote the
request back, so a failure is recorded as a short code, never as text.
"""

import asyncio
import contextlib
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

import redis.asyncio as redis

from llmproxy.config_model import UsageStreamParams
from llmproxy.core.logger import get_logger
from llmproxy.core.redis_utils import await_redis_result

logger = get_logger(__name__)

#: Ceiling for any single reported token count. Above this the number is not a
#: usage report, and a 64-bit consumer column could not hold it.
_MAX_TOKEN_COUNT = 2**63 - 1

#: Records buffered before new ones are dropped. Telemetry must never apply
#: back pressure to proxied traffic, so a saturated queue loses records.
_WRITE_QUEUE_SIZE = 10_000
#: A full queue logs once per this many dropped records rather than per record.
_DROP_LOG_INTERVAL = 100

#: Caller header values are copied verbatim, so a hostile or buggy caller
#: cannot be allowed to push arbitrarily large strings into the stream.
_MAX_CALLER_HEADER_LENGTH = 256
#: Error labels are codes and exception class names; this only bounds them.
_MAX_ERROR_LENGTH = 128

#: Reasoning efforts are short lowercase words ("minimal", "high"). The value
#: comes from the request body, so anything else is dropped rather than copied
#: into a label.
_REASONING_EFFORT = re.compile(r"[a-z_]{1,16}")
#: The shape of a provider error code or type, e.g. ``content_filter``.
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")

#: Every stream event the observer reads contains one of these; anything else
#: is a delta, skipped without being parsed.
_PARSED_MARKERS = ('"usage"', '"error"', "response.failed")

#: A stream the client abandoned. Nothing failed upstream, and folding these
#: into the error rate would make a user closing a tab look like an outage.
_CLIENT_CLOSED_STATUS = 499
#: A stream that raised part-way through.
_STREAM_ERROR_STATUS = 500
#: A stream the upstream answered with 200 and then reported a failure inside.
_UPSTREAM_STREAM_ERROR_STATUS = 502

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
    """Extract the usage report from a replayed SSE stream.

    A stream reports usage once, at its end, so the scan starts there. It runs
    on the event loop before the replay begins, and a chat stream whose every
    chunk carries `"usage": null` would otherwise be parsed in full.
    """
    observer = StreamUsageObserver()
    for chunk in reversed(chunks):
        observer.observe(chunk)
        if observer.usage is not None:
            break
    return observer.usage


def extract_reasoning_effort(request_data: Mapping[str, Any]) -> Optional[str]:
    """Read the requested reasoning effort from either API surface."""
    reasoning = request_data.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    if effort is None:
        effort = request_data.get("reasoning_effort")
    if isinstance(effort, str) and _REASONING_EFFORT.fullmatch(effort):
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
            collected[name.lower()] = value.strip()[:_MAX_CALLER_HEADER_LENGTH]
    return collected


def error_code(detail: Any) -> Optional[str]:
    """Return the error code or type an upstream error body carries, if any.

    Accepts the raw body text or an already-parsed error object. Only the code
    is ever recorded, never the message: error bodies can quote the request
    back, and content-filter rejections do.
    """
    if isinstance(detail, (str, bytes)):
        try:
            detail = json.loads(detail)
        except ValueError:
            return None
    if isinstance(detail, Mapping) and isinstance(detail.get("error"), Mapping):
        detail = detail["error"]
    if not isinstance(detail, Mapping):
        return None
    for key in ("code", "type"):
        value = detail.get(key)
        if isinstance(value, str) and _ERROR_CODE.fullmatch(value):
            return value
    return None


def _sse_payloads(chunk: str) -> Iterator[Mapping[str, Any]]:
    """Yield the JSON objects carried by a chunk's ``data:`` lines."""
    for line in chunk.splitlines():
        if not line.startswith("data:"):
            continue
        # The single space after the colon is optional in the SSE spec.
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except ValueError:
            continue
        if isinstance(parsed, Mapping):
            yield parsed


def _stream_failure(payload: Mapping[str, Any]) -> Optional[str]:
    """Return an error code when an SSE event reports a failure, else None.

    An upstream that has already answered 200 can only report a failure inside
    the stream: Chat Completions sends an ``error`` object, Responses a
    ``response.failed`` or ``error`` event.
    """
    if payload.get("type") == "response.failed":
        response = payload.get("response")
        detail = response.get("error") if isinstance(response, Mapping) else None
        return error_code(detail) or "response_failed"
    if payload.get("type") == "error" or payload.get("error"):
        return error_code(payload) or "stream_error"
    return None


class StreamUsageObserver:
    """Scan SSE chunks for the terminal usage report and in-band failures.

    Both API surfaces report usage exactly once, at the end: Responses inside
    the ``response.completed`` event, Chat Completions in a final chunk carrying
    an empty ``choices`` array. Parsing every delta chunk as JSON would be
    costly on long streams, so a substring check gates the parse.
    """

    def __init__(self) -> None:
        self.usage: Optional[Dict[str, int]] = None
        self.error: Optional[str] = None

    def observe(self, chunk: str) -> None:
        if not chunk or not any(marker in chunk for marker in _PARSED_MARKERS):
            return
        for payload in _sse_payloads(chunk):
            usage = normalize_usage(payload)
            if usage is not None:
                self.usage = usage
            failure = _stream_failure(payload)
            if failure is not None:
                self.error = failure


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

    @classmethod
    def from_request(
        cls,
        *,
        api_surface: str,
        model_group: str,
        request_data: Mapping[str, Any],
        caller: Dict[str, str],
        start_time: float,
    ) -> "UsageContext":
        return cls(
            api_surface=api_surface,
            model_group=model_group,
            streaming=bool(request_data.get("stream", False)),
            reasoning_effort=extract_reasoning_effort(request_data),
            caller=caller,
            start_time=start_time,
        )

    def latency_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)


@dataclass(frozen=True)
class ServedBy:
    """The endpoint an upstream call went to; empty when none was reached."""

    endpoint_model: Optional[str] = None
    endpoint_id: Optional[str] = None
    endpoint_base_url: Optional[str] = None

    @classmethod
    def from_response(cls, response: Mapping[str, Any]) -> "ServedBy":
        return cls(
            endpoint_model=response.get("endpoint_model"),
            endpoint_id=response.get("endpoint_id"),
            endpoint_base_url=response.get("endpoint_base_url"),
        )


def _is_client_abort(exc: BaseException) -> bool:
    """Whether the consumer went away rather than the upstream failing."""
    return isinstance(exc, (asyncio.CancelledError, GeneratorExit))


def _as_text(chunk: Union[str, bytes, bytearray]) -> str:
    if isinstance(chunk, (bytes, bytearray)):
        return chunk.decode("utf-8", errors="ignore")
    return chunk


@dataclass(frozen=True)
class _StreamSink:
    """Where records go: the Redis client and the stream it appends to."""

    client: redis.Redis
    params: UsageStreamParams

    async def append(self, record: Dict[str, Any]) -> None:
        await await_redis_result(
            self.client.xadd(
                self.params.stream_key,
                {"payload": json.dumps(record)},
                maxlen=self.params.max_len,
                approximate=True,
            )
        )


class UsageRecorder:
    """Queue usage records for a background writer.

    ``record`` is deliberately synchronous. It is called from ``finally``
    blocks that can already be inside a cancelled scope -- a client that
    disconnects mid-stream is the common case -- and any ``await`` there raises
    ``CancelledError`` before the write happens, silently losing the record for
    a stream the upstream has already billed. Handing the record to a queue
    cannot be cancelled and cannot raise.
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        params: Optional[UsageStreamParams],
    ) -> None:
        self._sink = (
            _StreamSink(redis_client, params)
            if redis_client is not None and params is not None and params.enabled
            else None
        )
        self._queue: Optional["asyncio.Queue[Dict[str, Any]]"] = None
        self._task: Optional["asyncio.Task[None]"] = None
        self._closed = False
        self._dropped = 0

    @classmethod
    def disabled(cls) -> "UsageRecorder":
        """A recorder that writes nothing, for when telemetry is not configured."""
        return cls(None, None)

    @property
    def enabled(self) -> bool:
        return self._sink is not None

    def caller_headers(self, headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
        """Return the configured attribution headers present on a request."""
        if self._sink is None or not headers:
            return {}
        return extract_caller_headers(headers, self._sink.params.caller_headers)

    def record(
        self,
        context: UsageContext,
        *,
        status_code: int,
        attempts: int,
        served_by: Optional[ServedBy] = None,
        usage: Optional[Dict[str, int]] = None,
        cache_hit: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Queue one record for a finished request. Never blocks or raises."""
        sink = self._sink
        if sink is None or self._closed:
            return

        record: Dict[str, Any] = {
            "request_id": context.request_id,
            "recorded_at": time.time(),
            "api_surface": context.api_surface,
            "model_group": context.model_group,
            **asdict(served_by or ServedBy()),
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
            record["error"] = error[:_MAX_ERROR_LENGTH]

        self._enqueue(sink, record)

    def observe_stream(
        self,
        stream: AsyncIterator[Any],
        context: UsageContext,
        *,
        attempts: int,
        served_by: ServedBy,
    ) -> AsyncIterator[Any]:
        """Wrap a stream so one record is written when it ends, however it ends.

        Returns the stream untouched when telemetry is off, so streaming pays
        no inspection cost in the default configuration.
        """
        if self._sink is None:
            return stream
        return self._observed(stream, context, attempts, served_by)

    async def _observed(
        self,
        stream: AsyncIterator[Any],
        context: UsageContext,
        attempts: int,
        served_by: ServedBy,
    ) -> AsyncGenerator[Any, None]:
        observer = StreamUsageObserver()
        status_code = 200
        error: Optional[str] = None
        try:
            async for chunk in stream:
                observer.observe(_as_text(chunk))
                yield chunk
        except BaseException as exc:
            # A stream that dies mid-flight still consumed upstream tokens, so
            # the partial observation is recorded with a failure status.
            # BaseException rather than Exception: a client disconnect arrives
            # as CancelledError or GeneratorExit.
            status_code = (
                _CLIENT_CLOSED_STATUS if _is_client_abort(exc) else _STREAM_ERROR_STATUS
            )
            error = type(exc).__name__
            raise
        finally:
            if error is None and observer.error is not None:
                status_code, error = _UPSTREAM_STREAM_ERROR_STATUS, observer.error
            self.record(
                context,
                status_code=status_code,
                attempts=attempts,
                served_by=served_by,
                usage=observer.usage,
                error=error,
            )

    def _enqueue(self, sink: _StreamSink, record: Dict[str, Any]) -> None:
        try:
            queue = self._ensure_worker(sink)
        except RuntimeError:
            # No running loop (synchronous teardown); nothing can be written.
            return
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % _DROP_LOG_INTERVAL == 1:
                logger.warning("usage_record_queue_full", dropped=self._dropped)

    def _ensure_worker(self, sink: _StreamSink) -> "asyncio.Queue[Dict[str, Any]]":
        loop = asyncio.get_running_loop()
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=_WRITE_QUEUE_SIZE)
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._drain(self._queue, sink))
        return self._queue

    async def _drain(
        self, queue: "asyncio.Queue[Dict[str, Any]]", sink: _StreamSink
    ) -> None:
        while True:
            record = await queue.get()
            try:
                await sink.append(record)
            except Exception as exc:
                logger.warning(
                    "usage_record_write_failed",
                    stream_key=sink.params.stream_key,
                    error=str(exc),
                )
            finally:
                queue.task_done()

    async def flush(self, timeout: float = 2.0) -> None:
        """Wait for queued records to be written."""
        if self._queue is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("usage_record_flush_timeout", pending=self._queue.qsize())

    async def aclose(self, timeout: float = 2.0) -> None:
        """Write what is queued, then stop the writer.

        Records arriving afterwards are dropped rather than starting a new
        writer against a Redis connection that is about to close.
        """
        self._closed = True
        await self.flush(timeout)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
