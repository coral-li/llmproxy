import json

import pytest
import redis.asyncio as redis

from llmproxy.core.cache_manager import CacheManager, StreamingCacheWriter


@pytest.mark.asyncio
async def test_invalidate_request_removes_response_stream_keys():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    cache = CacheManager(r)
    request = {"model": "gpt-3.5-turbo", "input": "test"}
    key_base = cache._generate_cache_key(request)

    # Seed fake cache entries
    await r.set(key_base, json.dumps({"foo": "bar"}))
    await r.rpush(f"{key_base}:responses_stream", "chunk1")
    await r.set(f"{key_base}:responses_stream:normalized", json.dumps([{"event": "x"}]))

    assert await r.exists(key_base)
    assert await r.exists(f"{key_base}:responses_stream")
    assert await r.exists(f"{key_base}:responses_stream:normalized")

    # Invalidate request
    result = await cache.invalidate_request(request)
    assert result is True

    assert not await r.exists(key_base)
    assert not await r.exists(f"{key_base}:responses_stream")
    assert not await r.exists(f"{key_base}:responses_stream:normalized")

    await r.aclose()


def test_sse_parsing_handles_data_before_event():
    """Test that SSE parsing works correctly when data lines appear before event lines"""
    cache_manager = CacheManager(
        redis.Redis(), cache_enabled=False
    )  # Mock for parsing test
    writer = StreamingCacheWriter(cache_manager, {}, is_responses_api=True)

    # Test case 1: Normal order (event first)
    chunk_normal = """event: response.created
data: {"response": {"model": "gpt-4", "created": 1234567890}}

"""
    result_normal = writer._parse_responses_event(chunk_normal)
    assert result_normal is not None
    assert result_normal.event_type == "response.created"
    assert result_normal.data_type == "response"

    # Test case 2: Data first (this was the bug)
    chunk_data_first = """data: {"response": {"model": "gpt-4", "created": 1234567890}}
event: response.created

"""
    result_data_first = writer._parse_responses_event(chunk_data_first)
    assert result_data_first is not None
    assert result_data_first.event_type == "response.created"
    assert result_data_first.data_type == "response"

    # Test case 3: Multiple data lines (edge case)
    chunk_multiple_data = """event: response.output_text.delta
data: {"delta": "Hello", "output_index": 0}
data: {"delta": " world", "output_index": 0}

"""
    result_multiple = writer._parse_responses_event(chunk_multiple_data)
    assert result_multiple is not None
    assert result_multiple.event_type == "response.output_text.delta"
    assert result_multiple.content == "Hello"  # Should use first valid data line

    # Test case 4: Invalid JSON should be skipped
    chunk_invalid_json = """event: response.created
data: invalid-json
data: {"response": {"model": "gpt-4"}}

"""
    result_invalid = writer._parse_responses_event(chunk_invalid_json)
    assert result_invalid is not None
    assert result_invalid.event_type == "response.created"
