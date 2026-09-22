"""Tests for per-request usage telemetry."""

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest
import redis as sync_redis
import requests
from fastapi.responses import StreamingResponse

from llmproxy.clients.llm_client import LLMClient
from llmproxy.config_model import UsageStreamParams
from llmproxy.core.usage_telemetry import (
    StreamUsageObserver,
    UsageContext,
    UsageRecorder,
    extract_caller_headers,
    extract_reasoning_effort,
    normalize_usage,
)
from llmproxy.models.endpoint import Endpoint


class FakeRedis:
    """Minimal stand-in capturing XADD calls."""

    def __init__(self, fail: bool = False) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.fail = fail

    async def xadd(
        self,
        stream_key: str,
        fields: Dict[str, str],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        if self.fail:
            raise ConnectionError("redis is down")
        self.entries.append(
            {
                "stream_key": stream_key,
                "fields": fields,
                "maxlen": maxlen,
                "approximate": approximate,
            }
        )
        return "1-0"

    def records(self) -> List[dict]:
        return [json.loads(entry["fields"]["payload"]) for entry in self.entries]


def make_recorder(fail: bool = False) -> tuple:
    redis = FakeRedis(fail=fail)
    recorder = UsageRecorder(redis, UsageStreamParams())
    return recorder, redis


def make_context(**overrides: Any) -> UsageContext:
    defaults: Dict[str, Any] = {
        "api_surface": "responses",
        "model_group": "gpt-5",
        "streaming": False,
    }
    defaults.update(overrides)
    return UsageContext(**defaults)


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
        assert observer.endpoint_model == "gpt-5"

    def test_captures_chat_usage_chunk(self):
        observer = StreamUsageObserver()
        observer.observe(
            'data: {"model":"gpt-4.1","choices":[],'
            '"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\n'
        )
        assert observer.usage is not None
        assert observer.usage["input_tokens"] == 7
        assert observer.endpoint_model == "gpt-4.1"

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


class TestUsageRecorder:
    @pytest.mark.asyncio
    async def test_disabled_without_params(self):
        recorder = UsageRecorder(FakeRedis(), None)
        assert recorder.enabled is False
        recorder.record(make_context(), status_code=200)

    @pytest.mark.asyncio
    async def test_disabled_when_flag_off(self):
        redis = FakeRedis()
        recorder = UsageRecorder(redis, UsageStreamParams(enabled=False))
        recorder.record(make_context(), status_code=200)
        await recorder.flush()
        assert redis.entries == []

    @pytest.mark.asyncio
    async def test_writes_record_with_endpoint_and_usage(self):
        recorder, redis = make_recorder()
        endpoint = Endpoint(
            model="gpt-5",
            weight=1,
            params={"base_url": "https://example.openai.azure.com", "api_key": "k"},
        )
        recorder.record(
            make_context(caller={"x-coral-agent": "classifier"}),
            status_code=200,
            endpoint=endpoint,
            usage={"input_tokens": 3, "output_tokens": 1},
            attempts=2,
        )

        await recorder.flush()
        (record,) = redis.records()
        assert record["model_group"] == "gpt-5"
        assert record["endpoint_model"] == "gpt-5"
        assert record["endpoint_id"] == endpoint.id
        assert record["endpoint_base_url"] == "https://example.openai.azure.com"
        assert record["caller"] == {"x-coral-agent": "classifier"}
        assert record["usage"]["input_tokens"] == 3
        assert record["attempts"] == 2
        assert record["status_code"] == 200
        assert record["cache_hit"] is False
        assert isinstance(record["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_applies_stream_bounds(self):
        recorder, redis = make_recorder()
        recorder.record(make_context(), status_code=200)
        await recorder.flush()
        entry = redis.entries[0]
        assert entry["stream_key"] == "llmproxy:usage"
        assert entry["maxlen"] == 1_000_000
        assert entry["approximate"] is True

    @pytest.mark.asyncio
    async def test_redis_failure_is_swallowed(self):
        """Telemetry must never turn a healthy proxied request into an error."""
        recorder, _ = make_recorder(fail=True)
        recorder.record(make_context(), status_code=200)

    @pytest.mark.asyncio
    async def test_error_is_truncated(self):
        recorder, redis = make_recorder()
        recorder.record(make_context(), status_code=500, error="x" * 5000)
        await recorder.flush()
        (record,) = redis.records()
        assert len(record["error"]) == 500

    def test_build_context_reads_request(self):
        recorder, _ = make_recorder()
        context = recorder.build_context(
            api_surface="responses",
            model_group="gpt-5",
            request_data={"stream": True, "reasoning": {"effort": "high"}},
            caller={"x-coral-agent": "a"},
        )
        assert context.streaming is True
        assert context.reasoning_effort == "high"
        assert context.caller == {"x-coral-agent": "a"}

    def test_caller_headers_empty_when_disabled(self):
        recorder = UsageRecorder(None, None)
        assert recorder.caller_headers({"x-coral-agent": "a"}) == {}


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


class TestStreamObservationInHandler:
    """The handler wrapper must record once the stream drains, and pass bytes through."""

    @pytest.mark.asyncio
    async def test_records_after_stream_completes(self):
        from llmproxy.api.chat_completions import ChatCompletionHandler
        from llmproxy.core.usage_telemetry import current_usage_context

        recorder, redis = make_recorder()
        handler = ChatCompletionHandler.__new__(ChatCompletionHandler)
        handler.usage_recorder = recorder

        endpoint = Endpoint(model="gpt-4.1", weight=1, params={"api_key": "k"})
        context = make_context(api_surface="chat", streaming=True)
        current_usage_context.set(context)

        async def source() -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield (
                b'data: {"choices":[],"usage":'
                b'{"prompt_tokens":6,"completion_tokens":2}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        wrapped = handler._observe_stream_usage(source(), endpoint, {"attempts": 1})

        chunks = [chunk async for chunk in wrapped]
        assert len(chunks) == 3
        assert all(isinstance(chunk, bytes) for chunk in chunks)

        await recorder.flush()
        (record,) = redis.records()
        assert record["usage"]["input_tokens"] == 6
        assert record["usage"]["output_tokens"] == 2
        assert record["status_code"] == 200
        assert record["streaming"] is True

    @pytest.mark.asyncio
    async def test_records_partial_usage_when_stream_fails(self):
        from llmproxy.api.chat_completions import ChatCompletionHandler
        from llmproxy.core.usage_telemetry import current_usage_context

        recorder, redis = make_recorder()
        handler = ChatCompletionHandler.__new__(ChatCompletionHandler)
        handler.usage_recorder = recorder

        endpoint = Endpoint(model="gpt-4.1", weight=1, params={"api_key": "k"})
        current_usage_context.set(make_context(api_surface="chat", streaming=True))

        async def failing_source() -> AsyncIterator[str]:
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            raise RuntimeError("upstream dropped")

        wrapped = handler._observe_stream_usage(
            failing_source(), endpoint, {"attempts": 1}
        )

        with pytest.raises(RuntimeError):
            async for _ in wrapped:
                pass

        await recorder.flush()
        (record,) = redis.records()
        assert record["status_code"] == 500
        assert record["error"] == "RuntimeError: upstream dropped"

    @pytest.mark.asyncio
    async def test_stream_untouched_when_telemetry_disabled(self):
        from llmproxy.api.chat_completions import ChatCompletionHandler

        handler = ChatCompletionHandler.__new__(ChatCompletionHandler)
        handler.usage_recorder = UsageRecorder(None, None)

        endpoint = Endpoint(model="gpt-4.1", weight=1, params={"api_key": "k"})

        async def source() -> AsyncIterator[str]:
            yield "data: x\n\n"

        original = source()
        assert handler._observe_stream_usage(original, endpoint, {}) is original


# ---------------------------------------------------------------------------
# End-to-end: a real request through the running proxy must land in the stream.
# ---------------------------------------------------------------------------

TEST_STREAM_KEY = "llmproxy:test:usage"


def read_records_for_agent(agent: str, limit: int = 2000) -> List[dict]:
    """Return usage records emitted for one agent label."""
    client = sync_redis.Redis(host="localhost", port=6379, decode_responses=True)
    try:
        entries = client.xrevrange(TEST_STREAM_KEY, count=limit)
    finally:
        client.close()

    records = []
    for _entry_id, fields in entries:
        try:
            record = json.loads(fields["payload"])
        except (KeyError, json.JSONDecodeError):
            continue
        if record.get("caller", {}).get("x-coral-agent") == agent:
            records.append(record)
    return records


class TestUsageTelemetryEndToEnd:
    def test_non_streaming_request_is_recorded(self, proxy_url, model):
        agent = f"agent-{uuid.uuid4().hex[:8]}"

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

        records = read_records_for_agent(agent)
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

    def test_streaming_request_records_usage_from_final_chunk(self, proxy_url, model):
        agent = f"agent-{uuid.uuid4().hex[:8]}"

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
            body = response.text

        assert "[DONE]" in body

        records = read_records_for_agent(agent)
        assert len(records) == 1, f"expected one record, got {records}"
        record = records[0]

        assert record["streaming"] is True
        assert record["status_code"] == 200
        # Proves the usage chunk survived the proxy's chunk filter.
        assert record["usage"] is not None
        assert record["usage"]["input_tokens"] == 10
        assert record["usage"]["output_tokens"] > 0

    def test_cache_hit_is_recorded_separately(self, proxy_url, model):
        agent = f"agent-{uuid.uuid4().hex[:8]}"
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

        records = read_records_for_agent(agent)
        assert len(records) == 2

        cache_hits = [record for record in records if record["cache_hit"]]
        assert len(cache_hits) == 1
        # A replayed response consumed no upstream call, so no endpoint is named.
        assert cache_hits[0]["attempts"] == 0
        assert cache_hits[0]["endpoint_id"] is None

    def test_request_without_headers_still_recorded(self, proxy_url, model):
        """Attribution is optional; an unlabelled call must still be counted."""
        marker = f"unlabelled {uuid.uuid4().hex[:8]}"
        response = requests.post(
            f"{proxy_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            timeout=30,
        )
        assert response.status_code == 200

        client = sync_redis.Redis(host="localhost", port=6379, decode_responses=True)
        try:
            entries = client.xrevrange(TEST_STREAM_KEY, count=50)
        finally:
            client.close()

        unlabelled = [
            json.loads(fields["payload"])
            for _entry_id, fields in entries
            if json.loads(fields["payload"]).get("caller") == {}
        ]
        assert unlabelled, "expected at least one record without caller headers"


class TestAuditFixes:
    """Regression cover for defects the audit found."""

    @pytest.mark.asyncio
    async def test_client_disconnect_is_still_recorded(self):
        """Starlette cancels the scope; an awaited write would be dropped."""
        from llmproxy.api.chat_completions import ChatCompletionHandler
        from llmproxy.core.usage_telemetry import current_usage_context

        recorder, redis = make_recorder()
        handler = ChatCompletionHandler.__new__(ChatCompletionHandler)
        handler.usage_recorder = recorder
        endpoint = Endpoint(model="gpt-4.1", weight=1, params={"api_key": "k"})
        current_usage_context.set(make_context(api_surface="chat", streaming=True))

        async def source():
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":4}}\n\n'

        wrapped = handler._observe_stream_usage(source(), endpoint, {"attempts": 1})
        async for _ in wrapped:
            break
        await wrapped.aclose()
        await recorder.flush()

        (record,) = redis.records()
        # 499: the consumer went away, the upstream did not fail.
        assert record["status_code"] == 499
        assert "GeneratorExit" in record["error"]

    @pytest.mark.asyncio
    async def test_telemetry_cannot_fail_a_healthy_request(self):
        """json.loads decodes 1e999 to inf, and int() used to raise on it."""
        from llmproxy.api.chat_completions import ChatCompletionHandler

        recorder, redis = make_recorder()
        handler = ChatCompletionHandler.__new__(ChatCompletionHandler)
        handler.usage_recorder = recorder
        context = make_context()

        hostile = json.loads(
            '{"usage": {"prompt_tokens": 1e999, ' '"completion_tokens": NaN}}'
        )
        handler._record_completed_response(
            context, {"status_code": 200, "data": hostile}
        )
        await recorder.flush()

        (record,) = redis.records()
        assert record["status_code"] == 200
        assert record["usage"]["input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_exhaustion_records_the_real_attempt_count(self):
        from llmproxy.api.chat_completions import ChatCompletionHandler

        handler = ChatCompletionHandler.__new__(ChatCompletionHandler)
        exhausted = handler._all_endpoints_failed_response(2)
        none_available = handler._no_endpoint_response("m")

        assert exhausted["attempts"] == 2
        # Nothing was tried, so reporting 1 would invent an upstream call.
        assert none_available["attempts"] == 0

    @pytest.mark.asyncio
    async def test_streamed_cache_hit_reports_what_the_cache_saved(self):
        from llmproxy.api.chat_completions import ChatCompletionHandler

        recorder, redis = make_recorder()
        handler = ChatCompletionHandler.__new__(ChatCompletionHandler)
        handler.usage_recorder = recorder
        context = make_context(streaming=True)
        context.cached_chunks = [
            'data: {"choices":[],"usage":{"prompt_tokens":11,'
            '"completion_tokens":3}}\n\n'
        ]

        handler._record_cache_hit(context, StreamingResponse(iter([])))
        await recorder.flush()

        (record,) = redis.records()
        assert record["cache_hit"] is True
        assert record["usage"]["input_tokens"] == 11

    def test_absurd_token_counts_are_clamped_not_stored(self):
        usage = normalize_usage({"usage": {"prompt_tokens": 10**40}})
        assert usage is not None
        assert usage["input_tokens"] <= 2**63 - 1
