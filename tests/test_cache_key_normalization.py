from unittest.mock import AsyncMock

import pytest

from llmproxy.core.cache_manager import CacheManager


@pytest.mark.asyncio
async def test_responses_api_cache_key_considers_0_and_0_0_equivalent():
    """Failing test: Two semantically identical requests should map to the same cache key.

    Some clients may serialize numeric fields like temperature as 0 vs 0.0 across
    identical calls. We still expect a cache hit on the second call.
    """

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    # First request has temperature as float 0.0
    request_a = {
        "model": "gpt-4.1",
        "input": "Hello",
        "temperature": 0.0,
    }
    response_obj = {
        "object": "response",
        "outputs": [{"type": "message", "content": [{"type": "text", "text": "Hi"}]}],
    }

    # Second request is identical but with integer 0
    request_b = {
        "model": "gpt-4.1",
        "input": "Hello",
        "temperature": 0,
    }

    # Write cache for first request
    await cache.set(request_a, response_obj)

    # Simulate a later read using the slightly different numeric representation
    await cache.get(request_b)

    # Expect a hit, indicating we treated the two requests equivalently
    # This assertion is on the Redis client interactions: on hit we expect a get call
    # with the SAME key as produced by request_a; since keys differ today, this will fail.
    # We assert that both get calls used the same underlying key by comparing the first
    # argument of get for each call.
    assert mock_redis.get.await_count >= 1, "Expected at least one Redis GET call"
    get_calls = [args[0][0] for args in mock_redis.get.await_args_list]
    # The last get corresponds to request_b
    # The setex key was computed from request_a earlier
    set_calls = [args[0][0] for args in mock_redis.setex.await_args_list]

    assert set_calls, "Expected a cache SETEX call for the first request"
    assert get_calls, "Expected a cache GET call for the second request"

    assert (
        get_calls[-1] == set_calls[-1]
    ), f"Keys differ for semantically identical requests: {get_calls[-1]} vs {set_calls[-1]}"
