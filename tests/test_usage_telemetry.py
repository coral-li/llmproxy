"""Tests for per-request usage telemetry."""

import asyncio
import json
import random
import string
import time
import uuid
from typing import Any, AsyncIterator, Callable, List

import anyio
import pytest
import redis as sync_redis
import requests
from fastapi import HTTPException
from handler_harness import (
    FakeRedis,
    cache_config,
    drain,
    general_settings,
    make_context,
    make_endpoint,
    make_handler,
    make_recorder,
    upstream_error,
)
from pydantic import ValidationError

from llmproxy.api.embeddings import EmbeddingHandler
from llmproxy.api.responses import ResponseHandler
from llmproxy.clients.llm_client import LLMClient
from llmproxy.config_model import LLMProxyConfig, UsageStreamParams
from llmproxy.core import usage_telemetry
from llmproxy.core.usage_telemetry import (
    ServedBy,
    StreamUsageObserver,
    UsageContext,
    UsageRecorder,
    error_code,
    extract_caller_headers,
    extract_reasoning_effort,
    normalize_usage,
    usage_from_chunks,
)

#: Every field a record carries; `error` is added only when the call failed.
RECORD_FIELDS = {
    "request_id",
    "recorded_at",
    "api_surface",
    "model_group",
    "endpoint_model",
    "endpoint_id",
    "endpoint_base_url",
    "streaming",
    "reasoning_effort",
    "status_code",
    "cache_hit",
    "attempts",
    "latency_ms",
    "caller",
    "usage",
}

CHAT_USAGE_CHUNK = (
    'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}\n\n'
)
#: A delta as upstreams send it under `stream_options.include_usage`.
CHAT_DELTA_CHUNK = 'data: {"choices":[{"delta":{"content":"a"}}],"usage":null}\n\n'


def ok(data: dict) -> dict:
    return {"status_code": 200, "headers": {}, "data": data}


class TestNormalizeUsage:
    def test_responses_api_shape(self):
        usage = normalize_usage(
            {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "total_tokens": 140,
                    "input_tokens_details": {"cached_tokens": 25},
                    "output_tokens_details": {"reasoning_tokens": 16},
                }
            }
        )
        assert usage == {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "cached_tokens": 25,
            "cache_write_tokens": 0,
            "reasoning_tokens": 16,
        }

    def test_chat_completions_shape(self):
        usage = normalize_usage(
            {
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                }
            }
        )
        assert usage is not None
        assert usage["input_tokens"] == 12
        assert usage["output_tokens"] == 8
        assert usage["cached_tokens"] == 4
        assert usage["reasoning_tokens"] == 3

    def test_nested_under_response_key(self):
        """Responses stream events wrap the body in `response`."""
        usage = normalize_usage(
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 5, "output_tokens": 2}},
            }
        )
        assert usage is not None
        assert usage["input_tokens"] == 5
        # Falls back to input+output when the provider omits total_tokens.
        assert usage["total_tokens"] == 7

    def test_embeddings_shape_has_no_output(self):
        usage = normalize_usage({"usage": {"prompt_tokens": 9, "total_tokens": 9}})
        assert usage is not None
        assert usage["input_tokens"] == 9
        assert usage["output_tokens"] == 0

    @pytest.mark.parametrize(
        "payload",
        [None, {}, {"usage": None}, {"usage": "nope"}, [], "text"],
    )
    def test_missing_usage_returns_none(self, payload):
        assert normalize_usage(payload) is None

    def test_negative_and_bool_values_are_ignored(self):
        usage = normalize_usage({"usage": {"input_tokens": -5, "output_tokens": True}})
        assert usage is not None
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

    def test_non_finite_counts_degrade_to_zero(self):
        """`json.loads` decodes 1e999 to inf and NaN to nan; neither may raise."""
        hostile = json.loads(
            '{"usage": {"prompt_tokens": 1e999, "completion_tokens": NaN}}'
        )
        usage = normalize_usage(hostile)
        assert usage is not None
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

    def test_absurd_token_counts_are_clamped(self):
        usage = normalize_usage({"usage": {"prompt_tokens": 10**40}})
        assert usage is not None
        assert usage["input_tokens"] == 2**63 - 1


class TestReasoningEffort:
    def test_responses_api_nested_effort(self):
        assert extract_reasoning_effort({"reasoning": {"effort": "high"}}) == "high"

    def test_chat_completions_flat_effort(self):
        assert extract_reasoning_effort({"reasoning_effort": "low"}) == "low"

    @pytest.mark.parametrize(
        "request_data",
        [{}, {"reasoning": {}}, {"reasoning": "high"}, {"reasoning_effort": ""}],
    )
    def test_absent_effort(self, request_data):
        assert extract_reasoning_effort(request_data) is None

    @pytest.mark.parametrize(
        "effort", ["\u0000", "HIGH", "high; drop", "x" * 17, "\ud800", 3]
    )
    def test_a_value_that_is_not_an_effort_is_dropped(self, effort):
        """The value is caller-controlled and lands in a label column."""
        assert extract_reasoning_effort({"reasoning_effort": effort}) is None


class TestCallerHeaders:
    def test_matches_case_insensitively(self):
        collected = extract_caller_headers(
            {"X-Coral-Agent": "posting_classifier"}, ["x-coral-agent"]
        )
        assert collected == {"x-coral-agent": "posting_classifier"}

    def test_ignores_unlisted_and_blank_headers(self):
        collected = extract_caller_headers(
            {"x-coral-agent": "  ", "x-secret": "value"},
            ["x-coral-agent", "x-coral-run-id"],
        )
        assert collected == {}

    def test_truncates_oversized_values(self):
        collected = extract_caller_headers(
            {"x-coral-agent": "a" * 5000}, ["x-coral-agent"]
        )
        assert len(collected["x-coral-agent"]) == 256


class TestErrorCode:
    def test_reads_the_code_from_an_error_body(self):
        body = '{"error": {"code": "content_filter", "message": "Prompt: secret"}}'
        assert error_code(body) == "content_filter"

    def test_falls_back_to_the_error_type(self):
        assert error_code({"error": {"type": "invalid_request_error"}}) == (
            "invalid_request_error"
        )

    @pytest.mark.parametrize(
        "detail",
        [
            None,
            "Internal server error",
            b"\xff not json",
            '{"error": "a bare string"}',
            '{"error": {"code": "has spaces and; punctuation"}}',
            '{"error": {"code": 429}}',
        ],
    )
    def test_anything_else_yields_no_code(self, detail):
        assert error_code(detail) is None


class TestStreamUsageObserver:
    def test_captures_responses_completed_event(self):
        observer = StreamUsageObserver()
        observer.observe("event: response.output_text.delta\n")
        observer.observe('data: {"type":"response.output_text.delta"}\n\n')
        observer.observe(
            'data: {"type":"response.completed","response":'
            '{"model":"gpt-5","usage":{"input_tokens":11,"output_tokens":4}}}\n\n'
        )
        assert observer.usage is not None
        assert observer.usage["input_tokens"] == 11
        assert observer.error is None

    def test_captures_chat_usage_chunk(self):
        observer = StreamUsageObserver()
        observer.observe(CHAT_USAGE_CHUNK)
        assert observer.usage is not None
        assert observer.usage["input_tokens"] == 11

    def test_ignores_done_sentinel_and_malformed_json(self):
        observer = StreamUsageObserver()
        observer.observe("data: [DONE]\n\n")
        observer.observe('data: {"usage": broken\n\n')
        assert observer.usage is None

    def test_last_usage_wins(self):
        observer = StreamUsageObserver()
        observer.observe('data: {"usage":{"input_tokens":1}}\n\n')
        observer.observe('data: {"usage":{"input_tokens":99}}\n\n')
        assert observer.usage is not None
        assert observer.usage["input_tokens"] == 99

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            (
                'data: {"error":{"message":"overloaded","code":"server_error"}}\n\n',
                "server_error",
            ),
            (
                'data: {"type":"response.failed","response":'
                '{"error":{"code":"rate_limit_exceeded","message":"slow down"}}}\n\n',
                "rate_limit_exceeded",
            ),
            ('data: {"type":"response.failed","response":{}}\n\n', "response_failed"),
            ('data: {"type":"error","message":"boom"}\n\n', "error"),
        ],
    )
    def test_captures_a_failure_reported_inside_the_stream(self, chunk, expected):
        observer = StreamUsageObserver()
        observer.observe(chunk)
        assert observer.error == expected

    def test_a_null_error_field_is_not_a_failure(self):
        observer = StreamUsageObserver()
        observer.observe(
            'data: {"type":"response.created","response":{"error":null}}\n\n'
        )
        assert observer.error is None

    def test_a_replayed_stream_reports_its_final_usage(self):
        """Every delta says `"usage": null` when the caller asked for usage."""
        chunks = [CHAT_DELTA_CHUNK] * 50 + [CHAT_USAGE_CHUNK, "data: [DONE]\n\n"]

        usage = usage_from_chunks(chunks)

        assert usage is not None
        assert usage["input_tokens"] == 11

    def test_a_replayed_stream_without_usage_reports_none(self):
        assert usage_from_chunks([CHAT_DELTA_CHUNK, "data: [DONE]\n\n"]) is None


class TestUsageStreamParams:
    def test_defaults_keep_the_stream_out_of_the_response_cache(self):
        params = UsageStreamParams()
        assert params.stream_key == "llmproxy-telemetry:usage"
        assert params.max_len == 100_000
        assert params.caller_headers == []

    def test_the_default_stream_key_sits_beside_the_default_namespace(self):
        """The two defaults differ only after the prefix, which is what ships."""
        general_settings(usage_stream=UsageStreamParams())
        general_settings(cache_params=cache_config(), usage_stream=UsageStreamParams())

    def test_rejects_a_stream_key_inside_the_response_cache(self):
        """Clearing the cache deletes everything under `llmproxy:`."""
        with pytest.raises(ValidationError):
            general_settings(
                usage_stream=UsageStreamParams(stream_key="llmproxy:usage")
            )

    def test_the_cache_namespace_is_the_one_configured(self):
        assert general_settings().cache_namespace == "llmproxy"
        assert (
            general_settings(cache_params=cache_config()).cache_namespace == "llmproxy"
        )
        custom = general_settings(cache_params=cache_config(namespace="responses"))
        assert custom.cache_namespace == "responses"

    def test_the_stream_is_kept_out_of_a_configured_namespace(self):
        with pytest.raises(ValidationError):
            general_settings(
                cache_params=cache_config(namespace="responses"),
                usage_stream=UsageStreamParams(stream_key="responses:usage"),
            )
        # Outside the configured namespace is fine, whatever the default was.
        general_settings(
            cache_params=cache_config(namespace="responses"),
            usage_stream=UsageStreamParams(stream_key="llmproxy:usage"),
        )


class TestUsageRecorder:
    def test_disabled_without_params(self):
        recorder = UsageRecorder(FakeRedis(), None)
        assert recorder.enabled is False
        recorder.record(make_context(), status_code=200, attempts=1)

    @pytest.mark.asyncio
    async def test_disabled_when_flag_off(self):
        redis = FakeRedis()
        recorder = UsageRecorder(redis, UsageStreamParams(enabled=False))
        recorder.record(make_context(), status_code=200, attempts=1)
        await recorder.flush()
        assert redis.entries == []

    def test_caller_headers_empty_when_disabled(self):
        recorder = UsageRecorder.disabled()
        assert recorder.caller_headers({"x-coral-agent": "a"}) == {}

    @pytest.mark.asyncio
    async def test_writes_record_with_endpoint_and_usage(self):
        recorder, redis = make_recorder()
        recorder.record(
            make_context(caller={"x-coral-agent": "classifier"}),
            status_code=200,
            attempts=2,
            served_by=ServedBy(
                endpoint_model="gpt-5",
                endpoint_id="e1",
                endpoint_base_url="https://example.openai.azure.com",
            ),
            usage={"input_tokens": 3, "output_tokens": 1},
        )

        await recorder.flush()
        (record,) = redis.records()
        assert record["model_group"] == "gpt-5"
        assert record["endpoint_model"] == "gpt-5"
        assert record["endpoint_id"] == "e1"
        assert record["endpoint_base_url"] == "https://example.openai.azure.com"
        assert record["caller"] == {"x-coral-agent": "classifier"}
        assert record["usage"]["input_tokens"] == 3
        assert record["attempts"] == 2
        assert record["status_code"] == 200
        assert record["cache_hit"] is False
        assert isinstance(record["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_record_carries_exactly_the_documented_fields(self):
        """Consumers parse these by name; a rename must fail here, not there."""
        recorder, redis = make_recorder()
        recorder.record(make_context(), status_code=200, attempts=1)
        recorder.record(make_context(), status_code=503, attempts=0, error="x")
        await recorder.flush()

        succeeded, failed = redis.records()
        assert set(succeeded) == RECORD_FIELDS
        assert set(failed) == RECORD_FIELDS | {"error"}

    @pytest.mark.asyncio
    async def test_applies_stream_bounds(self):
        recorder, redis = make_recorder()
        recorder.record(make_context(), status_code=200, attempts=1)
        await recorder.flush()
        entry = redis.entries[0]
        assert entry["stream_key"] == "llmproxy-telemetry:usage"
        assert entry["maxlen"] == 100_000
        assert entry["approximate"] is True

    @pytest.mark.asyncio
    async def test_error_label_is_bounded(self):
        recorder, redis = make_recorder()
        recorder.record(make_context(), status_code=500, attempts=1, error="x" * 5000)
        await recorder.flush()
        (record,) = redis.records()
        assert len(record["error"]) == 128

    @pytest.mark.asyncio
    async def test_a_failed_write_does_not_stop_later_records(self):
        """Telemetry must never turn a healthy proxied request into an error."""
        recorder, redis = make_recorder(FakeRedis(fail_first=1))
        recorder.record(make_context(), status_code=200, attempts=1)
        recorder.record(make_context(), status_code=201, attempts=1)
        await recorder.flush()
        assert [record["status_code"] for record in redis.records()] == [201]

    @pytest.mark.asyncio
    async def test_a_saturated_queue_drops_records_without_raising(self, monkeypatch):
        monkeypatch.setattr(usage_telemetry, "_WRITE_QUEUE_SIZE", 1)
        release = asyncio.Event()

        class SlowRedis(FakeRedis):
            async def xadd(self, *args: Any, **kwargs: Any) -> str:
                await release.wait()
                return await super().xadd(*args, **kwargs)

        recorder, redis = make_recorder(SlowRedis())
        for status_code in (200, 201, 202):
            recorder.record(make_context(), status_code=status_code, attempts=1)
        release.set()
        await recorder.flush()
        assert [record["status_code"] for record in redis.records()] == [200]

    @pytest.mark.asyncio
    async def test_aclose_writes_what_is_queued(self):
        recorder, redis = make_recorder()
        for _ in range(5):
            recorder.record(make_context(), status_code=200, attempts=1)
        await recorder.aclose()
        assert len(redis.records()) == 5

    @pytest.mark.asyncio
    async def test_records_after_close_are_dropped(self):
        recorder, redis = make_recorder()
        await recorder.aclose()
        recorder.record(make_context(), status_code=200, attempts=1)
        await recorder.flush()
        assert redis.records() == []

    @pytest.mark.asyncio
    async def test_a_hanging_redis_cannot_hold_up_shutdown(self):
        class HangingRedis(FakeRedis):
            async def xadd(self, *args: Any, **kwargs: Any) -> str:
                await asyncio.Event().wait()
                return ""

        recorder, _redis = make_recorder(HangingRedis())
        recorder.record(make_context(), status_code=200, attempts=1)

        await asyncio.wait_for(recorder.aclose(timeout=0.05), timeout=2)

    def test_a_record_with_no_running_loop_is_dropped_without_raising(self):
        """Synchronous teardown has no loop to hand the record to."""
        recorder, redis = make_recorder()
        recorder.record(make_context(), status_code=200, attempts=1)
        assert redis.entries == []

    def test_context_reads_the_request(self):
        context = UsageContext.from_request(
            api_surface="responses",
            model_group="gpt-5",
            request_data={"stream": True, "reasoning": {"effort": "high"}},
            caller={"x-coral-agent": "a"},
            start_time=time.time(),
        )
        assert context.streaming is True
        assert context.reasoning_effort == "high"
        assert context.caller == {"x-coral-agent": "a"}


class TestStreamObservation:
    """One record per stream, written when it ends however it ends."""

    SERVED_BY = ServedBy(endpoint_model="gpt-4.1", endpoint_id="e1")

    def observe(self, recorder: UsageRecorder, source: AsyncIterator[Any]) -> Any:
        return recorder.observe_stream(
            source,
            make_context(api_surface="chat", streaming=True),
            attempts=1,
            served_by=self.SERVED_BY,
        )

    @pytest.mark.asyncio
    async def test_records_usage_once_the_stream_completes(self):
        recorder, redis = make_recorder()

        async def source() -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield CHAT_USAGE_CHUNK.encode()
            yield b"data: [DONE]\n\n"

        chunks = await drain(self.observe(recorder, source()))
        assert len(chunks) == 3
        assert all(isinstance(chunk, bytes) for chunk in chunks)

        await recorder.flush()
        (record,) = redis.records()
        assert record["usage"]["input_tokens"] == 11
        assert record["status_code"] == 200
        assert record["streaming"] is True
        assert record["endpoint_id"] == "e1"
        assert "error" not in record

    @pytest.mark.asyncio
    async def test_a_stream_that_raises_records_partial_usage_as_a_failure(self):
        recorder, redis = make_recorder()

        async def source() -> AsyncIterator[str]:
            yield CHAT_USAGE_CHUNK
            raise RuntimeError("upstream dropped the connection")

        with pytest.raises(RuntimeError):
            await drain(self.observe(recorder, source()))

        await recorder.flush()
        (record,) = redis.records()
        assert record["status_code"] == 500
        assert record["usage"]["input_tokens"] == 11
        # The exception class, never its message.
        assert record["error"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_a_closed_stream_records_a_client_abort(self):
        recorder, redis = make_recorder()

        async def source() -> AsyncIterator[str]:
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            yield CHAT_USAGE_CHUNK

        wrapped = self.observe(recorder, source())
        async for _ in wrapped:
            break
        await wrapped.aclose()
        await recorder.flush()

        (record,) = redis.records()
        # 499: the consumer went away, the upstream did not fail.
        assert record["status_code"] == 499
        assert record["error"] == "GeneratorExit"

    @pytest.mark.asyncio
    async def test_a_disconnect_under_a_cancelled_scope_is_recorded(self):
        """A disconnect cancels the scope, where any await in `finally` raises."""
        recorder, redis = make_recorder()
        never = asyncio.Event()

        async def source() -> AsyncIterator[str]:
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            await never.wait()
            yield "unreachable"

        wrapped = self.observe(recorder, source())

        async def consume() -> None:
            await drain(wrapped)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(consume)
            await asyncio.sleep(0.01)
            task_group.cancel_scope.cancel()

        await recorder.flush()
        (record,) = redis.records()
        assert record["status_code"] == 499
        assert record["error"] == "CancelledError"

    @pytest.mark.asyncio
    async def test_a_failure_reported_inside_the_stream_is_not_a_success(self):
        recorder, redis = make_recorder()

        async def source() -> AsyncIterator[str]:
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            yield 'data: {"error":{"code":"server_error","message":"at token 9"}}\n\n'

        await drain(self.observe(recorder, source()))
        await recorder.flush()

        (record,) = redis.records()
        assert record["status_code"] == 502
        assert record["error"] == "server_error"

    def test_a_disabled_recorder_passes_the_stream_through_untouched(self):
        async def source() -> AsyncIterator[str]:
            yield "data: x\n\n"

        original = source()
        wrapped = UsageRecorder.disabled().observe_stream(
            original, make_context(), attempts=1, served_by=ServedBy()
        )
        assert wrapped is original


class TestAttemptsAndEndpointReporting:
    """What a request records about failover, driven through `handle_request`."""

    @pytest.mark.asyncio
    async def test_success_after_failover_names_the_endpoint_that_served(self):
        first, second = make_endpoint("a"), make_endpoint("b")
        harness = make_handler(endpoints=(first, second))
        harness.load_balancer.select_endpoint.side_effect = [first, second]
        harness.llm_client.create_chat_completion.side_effect = [
            upstream_error(500, "boom"),
            ok({"choices": [], "usage": {"prompt_tokens": 4}}),
        ]

        await harness.handler.handle_request({"model": "m", "messages": []})

        record = await harness.only_record()
        assert record["status_code"] == 200
        assert record["attempts"] == 2
        assert record["endpoint_id"] == second.id
        assert record["endpoint_base_url"] == "https://b"
        assert record["usage"]["input_tokens"] == 4

    @pytest.mark.asyncio
    async def test_a_streamed_request_that_failed_over_counts_both_attempts(self):
        first, second = make_endpoint("a"), make_endpoint("b")
        harness = make_handler(endpoints=(first, second))
        harness.load_balancer.select_endpoint.side_effect = [first, second]

        async def upstream() -> AsyncIterator[bytes]:
            yield CHAT_USAGE_CHUNK.encode()

        harness.llm_client.create_chat_completion.side_effect = [
            upstream_error(500, "boom"),
            upstream(),
        ]

        response = await harness.handler.handle_request(
            {"model": "m", "messages": [], "stream": True}
        )
        await drain(response.body_iterator)

        record = await harness.only_record()
        assert record["status_code"] == 200
        assert record["attempts"] == 2
        assert record["endpoint_id"] == second.id

    @pytest.mark.asyncio
    async def test_no_endpoint_available_records_zero_attempts(self):
        harness = make_handler()
        harness.load_balancer.select_endpoint.return_value = None

        with pytest.raises(HTTPException) as raised:
            await harness.handler.handle_request({"model": "m", "messages": []})

        assert raised.value.status_code == 503
        record = await harness.only_record()
        assert record["status_code"] == 503
        assert record["attempts"] == 0
        assert record["error"] == "no_available_endpoints"
        assert record["endpoint_id"] is None

    @pytest.mark.asyncio
    async def test_running_out_of_endpoints_after_a_failure_keeps_the_attempt(self):
        failed = make_endpoint("a")
        harness = make_handler(
            endpoints=(failed, make_endpoint("b"), make_endpoint("c"))
        )
        # The one tried endpoint fails, and the rest are cooling down.
        harness.load_balancer.select_endpoint.side_effect = [failed, None]
        harness.llm_client.create_chat_completion.return_value = upstream_error(
            500, "boom"
        )

        with pytest.raises(HTTPException) as raised:
            await harness.handler.handle_request({"model": "m", "messages": []})

        assert raised.value.status_code == 503
        record = await harness.only_record()
        assert record["attempts"] == 1
        assert record["endpoint_id"] == failed.id
        assert record["error"] == "http_500"

    @pytest.mark.asyncio
    async def test_an_upstream_failure_records_its_code_not_its_text(self):
        endpoints = (make_endpoint("a"), make_endpoint("b"), make_endpoint("c"))
        harness = make_handler(endpoints=endpoints)
        harness.load_balancer.select_endpoint.side_effect = list(endpoints)
        harness.llm_client.create_chat_completion.return_value = upstream_error(
            400,
            '{"error": {"code": "content_filter", "message": "Prompt: Jane Doe"}}',
        )

        with pytest.raises(HTTPException):
            await harness.handler.handle_request({"model": "m", "messages": []})

        record = await harness.only_record()
        assert record["status_code"] == 400
        assert record["attempts"] == 3
        assert record["error"] == "content_filter"
        assert "Jane Doe" not in json.dumps(record)

    @pytest.mark.asyncio
    async def test_an_attempt_that_raised_names_its_endpoint_and_count(self):
        endpoints = (make_endpoint("a"), make_endpoint("b"), make_endpoint("c"))
        harness = make_handler(endpoints=endpoints)
        harness.load_balancer.select_endpoint.side_effect = list(endpoints)
        harness.llm_client.create_chat_completion.side_effect = RuntimeError(
            "connect failed"
        )

        with pytest.raises(HTTPException):
            await harness.handler.handle_request({"model": "m", "messages": []})

        record = await harness.only_record()
        assert record["attempts"] == 3
        assert record["endpoint_base_url"] == "https://c"
        assert record["error"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_exhaustion_is_attributed_to_the_last_endpoint_tried(self):
        """With one endpoint per group every upstream failure ends this way."""
        only = make_endpoint("a")
        harness = make_handler(endpoints=(only,))
        harness.load_balancer.select_endpoint.return_value = only
        harness.llm_client.create_chat_completion.return_value = upstream_error(
            429, "slow down"
        )

        with pytest.raises(HTTPException) as raised:
            await harness.handler.handle_request({"model": "m", "messages": []})

        assert raised.value.status_code == 503
        record = await harness.only_record()
        assert record["status_code"] == 503
        assert record["attempts"] == 1
        assert record["endpoint_base_url"] == "https://a"
        assert record["error"] == "http_429"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 413, 422])
    async def test_a_refusal_on_the_only_endpoint_is_recorded_as_a_refusal(
        self, status_code
    ):
        only = make_endpoint("a")
        harness = make_handler(endpoints=(only,))
        harness.load_balancer.select_endpoint.return_value = only
        harness.llm_client.create_chat_completion.return_value = upstream_error(
            status_code, '{"error": {"code": "invalid_json_schema", "message": "no"}}'
        )

        with pytest.raises(HTTPException) as raised:
            await harness.handler.handle_request({"model": "m", "messages": []})

        assert raised.value.status_code == status_code
        record = await harness.only_record()
        assert record["status_code"] == status_code
        assert record["attempts"] == 1
        assert record["endpoint_base_url"] == "https://a"
        assert record["error"] == "invalid_json_schema"

    @pytest.mark.asyncio
    async def test_a_streaming_request_that_cannot_connect_is_recorded(self):
        endpoint = make_endpoint("a")
        harness = make_handler(endpoints=(endpoint,), retries=1)
        harness.load_balancer.select_endpoint.return_value = endpoint
        harness.llm_client.create_chat_completion.side_effect = RuntimeError(
            "connect failed"
        )

        with pytest.raises(HTTPException) as raised:
            await harness.handler.handle_request(
                {"model": "m", "messages": [], "stream": True}
            )

        assert raised.value.status_code == 500
        record = await harness.only_record()
        assert record["status_code"] == 500
        assert record["attempts"] == 1
        assert record["endpoint_id"] == endpoint.id
        assert record["error"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_hostile_usage_cannot_fail_a_healthy_request(self):
        endpoint = make_endpoint("a")
        harness = make_handler(endpoints=(endpoint,))
        harness.load_balancer.select_endpoint.return_value = endpoint
        harness.llm_client.create_chat_completion.return_value = ok(
            json.loads('{"choices": [], "usage": {"prompt_tokens": 1e999}}')
        )

        body = await harness.handler.handle_request({"model": "m", "messages": []})

        assert body["choices"] == []
        record = await harness.only_record()
        assert record["usage"]["input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_caller_headers_are_read_from_the_request(self):
        endpoint = make_endpoint("a")
        harness = make_handler(endpoints=(endpoint,), caller_headers=("x-coral-agent",))
        harness.load_balancer.select_endpoint.return_value = endpoint
        harness.llm_client.create_chat_completion.return_value = ok({"choices": []})

        await harness.handler.handle_request(
            {"model": "m", "messages": []}, {"X-Coral-Agent": "classifier"}
        )

        record = await harness.only_record()
        assert record["caller"] == {"x-coral-agent": "classifier"}

    @pytest.mark.asyncio
    async def test_embeddings_record_their_surface(self):
        endpoint = make_endpoint("a")
        harness = make_handler(EmbeddingHandler, endpoints=(endpoint,))
        harness.load_balancer.select_endpoint.return_value = endpoint
        harness.llm_client.create_embedding.return_value = ok(
            {"data": [], "usage": {"prompt_tokens": 9, "total_tokens": 9}}
        )

        await harness.handler.handle_request({"model": "m", "input": "x"})

        record = await harness.only_record()
        assert record["api_surface"] == "embeddings"
        assert record["usage"]["input_tokens"] == 9
        assert record["endpoint_id"] == endpoint.id

    @pytest.mark.asyncio
    async def test_a_responses_stream_records_the_configured_model(self):
        """Upstreams report a dated snapshot; the record keeps the deployment."""
        endpoint = make_endpoint("a", model="gpt-5")
        harness = make_handler(ResponseHandler, endpoints=(endpoint,))
        harness.load_balancer.select_endpoint.return_value = endpoint

        async def upstream() -> AsyncIterator[str]:
            yield "event: response.completed\n"
            yield (
                'data: {"type":"response.completed","response":'
                '{"model":"gpt-5-2025-08-07",'
                '"usage":{"input_tokens":21,"output_tokens":5}}}\n\n'
            )

        harness.llm_client.create_response.return_value = upstream()

        response = await harness.handler.handle_request(
            {"model": "m", "input": "hi", "stream": True}
        )
        await drain(response.body_iterator)

        record = await harness.only_record()
        assert record["api_surface"] == "responses"
        assert record["usage"]["input_tokens"] == 21
        assert record["endpoint_model"] == "gpt-5"


class TestPinnedFollowUps:
    """A Responses follow-up goes to the endpoint that holds its state."""

    @pytest.mark.asyncio
    async def test_retries_on_the_pinned_endpoint_are_counted(self):
        pinned = make_endpoint("a", model="gpt-5")
        harness = make_handler(ResponseHandler, endpoints=(pinned,))
        affinity = harness.handler.response_affinity_manager
        affinity.get_endpoint_id.return_value = pinned.id
        harness.llm_client.create_response.side_effect = [
            upstream_error(500, "boom"),
            ok({"output": [], "usage": {"input_tokens": 3}}),
        ]

        await harness.handler.handle_request(
            {"model": "m", "input": "hi", "previous_response_id": "resp_1"}
        )

        record = await harness.only_record()
        assert record["status_code"] == 200
        assert record["attempts"] == 2
        assert record["endpoint_id"] == pinned.id

    @pytest.mark.asyncio
    async def test_a_follow_up_that_cannot_be_routed_is_recorded(self):
        """Refused before any upstream call, which is why nothing else records it."""
        harness = make_handler(ResponseHandler, endpoints=(make_endpoint("a"),))
        harness.handler.response_affinity_manager.get_endpoint_id.return_value = None

        with pytest.raises(HTTPException) as raised:
            await harness.handler.handle_request(
                {"model": "m", "input": "hi", "previous_response_id": "resp_gone"}
            )

        assert raised.value.status_code == 409
        harness.llm_client.create_response.assert_not_called()
        record = await harness.only_record()
        assert record["status_code"] == 409
        assert record["attempts"] == 0
        assert record["error"] == "affinity_expired"
        assert record["endpoint_id"] is None


class TestCacheHitRecording:
    @pytest.mark.asyncio
    async def test_a_streamed_hit_reports_what_the_cache_saved(self):
        harness = make_handler(cache=True)
        harness.cache_manager.get_streaming.return_value = [
            CHAT_USAGE_CHUNK,
            "data: [DONE]\n\n",
        ]

        response = await harness.handler.handle_request(
            {"model": "m", "messages": [], "stream": True}
        )
        await drain(response.body_iterator)

        record = await harness.only_record()
        assert record["cache_hit"] is True
        assert record["attempts"] == 0
        assert record["usage"]["input_tokens"] == 11

    @pytest.mark.asyncio
    async def test_a_hit_reports_the_cached_usage(self):
        harness = make_handler(cache=True)
        harness.cache_manager.get.return_value = {
            "choices": [],
            "usage": {"prompt_tokens": 7},
        }

        await harness.handler.handle_request({"model": "m", "messages": []})

        record = await harness.only_record()
        assert record["cache_hit"] is True
        assert record["attempts"] == 0
        assert record["endpoint_id"] is None
        assert record["usage"]["input_tokens"] == 7

    @pytest.mark.asyncio
    async def test_a_responses_hit_is_recorded_once(self):
        harness = make_handler(ResponseHandler, cache=True)
        harness.cache_manager.get.return_value = {
            "output": [],
            "usage": {"input_tokens": 5},
        }
        harness.cache_manager.get_affinity.return_value = None

        await harness.handler.handle_request({"model": "m", "input": "hi"})

        record = await harness.only_record()
        assert record["api_surface"] == "responses"
        assert record["cache_hit"] is True
        assert record["usage"]["input_tokens"] == 5

    @pytest.mark.asyncio
    async def test_a_streamed_responses_hit_is_recorded_once(self):
        harness = make_handler(ResponseHandler, cache=True)
        harness.cache_manager.get_streaming.return_value = [
            "event: response.completed\n",
            'data: {"type":"response.completed","response":'
            '{"id":"resp_1","usage":{"input_tokens":8,"output_tokens":2}}}\n\n',
        ]
        harness.cache_manager.get_affinity.return_value = None

        response = await harness.handler.handle_request(
            {"model": "m", "input": "hi", "stream": True}
        )
        await drain(response.body_iterator)

        record = await harness.only_record()
        assert record["api_surface"] == "responses"
        assert record["cache_hit"] is True
        assert record["usage"]["input_tokens"] == 8


class TestUsageChunkForwarding:
    """The chat usage chunk has an empty choices array and must survive."""

    @pytest.fixture
    def client(self):
        return LLMClient()

    def test_usage_chunk_with_empty_choices_is_forwarded(self, client):
        line = (
            'data: {"id":"chatcmpl-1","choices":[],'
            '"usage":{"prompt_tokens":10,"completion_tokens":5}}'
        )
        assert client._filter_and_yield_chunk(line) == [line + "\n\n"]

    def test_empty_choices_without_usage_still_filtered(self, client):
        line = 'data: {"id":"chatcmpl-1","choices":[]}'
        assert client._filter_and_yield_chunk(line) == []


# ---------------------------------------------------------------------------
# End-to-end: a real request through the running proxy must land in the stream.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def usage_stream_key(proxy_config: LLMProxyConfig) -> str:
    """The stream the test proxy writes to."""
    usage_stream = proxy_config.general_settings.usage_stream
    assert usage_stream is not None
    return usage_stream.stream_key


def wait_for_records(
    stream_key: str,
    matches: Callable[[dict], bool],
    expected: int,
    timeout: float = 5.0,
) -> List[dict]:
    """Return the records matching one test's request, once all have landed.

    Records are written by a background task after the response is sent, so
    reading once straight after the request would race it.
    """
    client = sync_redis.Redis(host="localhost", port=6379, decode_responses=True)
    deadline = time.monotonic() + timeout
    try:
        while True:
            entries = client.xrevrange(stream_key, count=500)
            records = [
                record
                for record in (json.loads(fields["payload"]) for _, fields in entries)
                if matches(record)
            ]
            if len(records) >= expected or time.monotonic() > deadline:
                return records
            time.sleep(0.05)
    finally:
        client.close()


def labelled(agent: str) -> Callable[[dict], bool]:
    return lambda record: record.get("caller", {}).get("x-coral-agent") == agent


def unique_agent() -> str:
    return f"agent-{uuid.uuid4().hex[:8]}"


class TestUsageTelemetryEndToEnd:
    def test_non_streaming_request_is_recorded(
        self, proxy_url, model, usage_stream_key
    ):
        agent = unique_agent()

        response = requests.post(
            f"{proxy_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"hello {agent}"}],
                "reasoning_effort": "low",
                "cache": {"no-cache": True},
            },
            headers={"X-Coral-Agent": agent, "X-Coral-Run-Id": "run-1"},
            timeout=30,
        )
        assert response.status_code == 200

        records = wait_for_records(usage_stream_key, labelled(agent), expected=1)
        assert len(records) == 1, f"expected one record, got {records}"
        record = records[0]

        assert record["api_surface"] == "chat"
        assert record["model_group"] == model
        assert record["status_code"] == 200
        assert record["streaming"] is False
        assert record["cache_hit"] is False
        assert record["reasoning_effort"] == "low"
        assert record["caller"]["x-coral-run-id"] == "run-1"
        # The endpoint that actually served is only knowable at the proxy.
        assert record["endpoint_id"]
        assert record["endpoint_base_url"].startswith("http://localhost:800")
        assert record["usage"]["input_tokens"] == 10
        assert record["usage"]["output_tokens"] > 0
        assert record["latency_ms"] >= 0

    def test_streaming_request_records_usage_from_final_chunk(
        self, proxy_url, model, usage_stream_key
    ):
        agent = unique_agent()

        with requests.post(
            f"{proxy_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"stream {agent}"}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache": {"no-cache": True},
            },
            headers={"X-Coral-Agent": agent},
            stream=True,
            timeout=30,
        ) as response:
            assert response.status_code == 200
            body = response.text

        assert "[DONE]" in body

        records = wait_for_records(usage_stream_key, labelled(agent), expected=1)
        assert len(records) == 1, f"expected one record, got {records}"
        record = records[0]

        assert record["streaming"] is True
        assert record["status_code"] == 200
        # Proves the usage chunk survived the proxy's chunk filter.
        assert record["usage"] is not None
        assert record["usage"]["input_tokens"] == 10
        assert record["usage"]["output_tokens"] > 0

    def test_a_stream_that_did_not_ask_for_usage_records_none(
        self, proxy_url, model, usage_stream_key
    ):
        """Chat streams report usage only under `stream_options.include_usage`."""
        agent = unique_agent()

        with requests.post(
            f"{proxy_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"stream {agent}"}],
                "stream": True,
                "cache": {"no-cache": True},
            },
            headers={"X-Coral-Agent": agent},
            stream=True,
            timeout=30,
        ) as response:
            assert response.status_code == 200
            assert "[DONE]" in response.text

        records = wait_for_records(usage_stream_key, labelled(agent), expected=1)
        assert len(records) == 1, f"expected one record, got {records}"
        assert records[0]["status_code"] == 200
        assert records[0]["usage"] is None

    def test_cache_hit_is_recorded_separately(self, proxy_url, model, usage_stream_key):
        agent = unique_agent()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"cache me {agent}"}],
        }
        headers = {"X-Coral-Agent": agent}

        first = requests.post(
            f"{proxy_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        assert first.status_code == 200
        second = requests.post(
            f"{proxy_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        assert second.status_code == 200
        assert second.json().get("_proxy_cache_hit") is True

        records = wait_for_records(usage_stream_key, labelled(agent), expected=2)
        assert len(records) == 2

        cache_hits = [record for record in records if record["cache_hit"]]
        assert len(cache_hits) == 1
        # A replayed response consumed no upstream call, so no endpoint is named,
        # but it still reports the usage the cache saved.
        assert cache_hits[0]["attempts"] == 0
        assert cache_hits[0]["endpoint_id"] is None
        assert cache_hits[0]["usage"]["input_tokens"] == 10

    def test_request_without_headers_still_recorded(
        self, proxy_url, model, usage_stream_key
    ):
        """Attribution is optional; an unlabelled call must still be counted."""
        # The effort is the one request field a record carries, so a unique one
        # identifies this request's record among everyone else's.
        marker = "".join(random.choices(string.ascii_lowercase, k=16))
        response = requests.post(
            f"{proxy_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "unlabelled"}],
                "reasoning_effort": marker,
                "cache": {"no-cache": True},
            },
            timeout=30,
        )
        assert response.status_code == 200

        records = wait_for_records(
            usage_stream_key,
            lambda record: record.get("reasoning_effort") == marker,
            expected=1,
        )
        assert len(records) == 1
        assert records[0]["caller"] == {}
        assert records[0]["api_surface"] == "chat"
