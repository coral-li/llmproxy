import json

import pytest
import redis.asyncio as redis

from llmproxy.core.cache_manager import CacheManager


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
