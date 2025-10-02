from unittest.mock import AsyncMock

import pytest

from llmproxy.core.cache_manager import (
    CacheManager,
    StreamingCacheAborted,
)


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Override global clear_cache fixture to avoid starting proxy server."""
    yield


@pytest.mark.asyncio
async def test_streaming_cache_writer_aborts_on_error_chunk():
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.setex = AsyncMock()
    mock_redis.rpush = AsyncMock()
    mock_redis.expire = AsyncMock()

    request = {"model": "gpt-3.5-turbo", "messages": [], "stream": True}
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)
    writer = await cache.create_streaming_cache_writer(request)

    normal_chunk = 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
    await writer.write_and_yield(normal_chunk)

    mock_redis.rpush.assert_awaited()
    sentinel_key = writer._error_sentinel_key()
    # First deletion clears any lingering failure sentinel before caching starts.
    first_delete_args = mock_redis.delete.await_args_list[0].args
    assert sentinel_key in first_delete_args

    error_chunk = 'data: {"error": {"message": "boom"}}'
    with pytest.raises(StreamingCacheAborted) as exc_info:
        await writer.write_and_yield(error_chunk)

    assert exc_info.value.chunk == error_chunk

    # Cleanup should remove cached stream keys and set the failure sentinel.
    cleanup_args = mock_redis.delete.await_args_list[-1].args
    assert writer._stream_key() in cleanup_args
    mock_redis.setex.assert_any_call(sentinel_key, cache.ttl, "1")


@pytest.mark.asyncio
async def test_intercept_stream_stops_after_error_chunk():
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.setex = AsyncMock()
    mock_redis.rpush = AsyncMock()
    mock_redis.expire = AsyncMock()

    request = {"model": "gpt-3.5-turbo", "messages": [], "stream": True}
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)
    writer = await cache.create_streaming_cache_writer(request)

    normal_chunk = 'data: {"choices": [{"delta": {"content": "Hi"}}]}'
    error_chunk = 'data: {"error": {"message": "boom"}}'
    extra_chunk = 'data: {"choices": [{"delta": {"content": "ignored"}}]}'

    async def upstream():
        yield normal_chunk
        yield error_chunk
        yield extra_chunk

    collected = []
    async for chunk in writer.intercept_stream(upstream()):
        collected.append(chunk)

    assert collected == [normal_chunk, error_chunk]
    # Only the first chunk should be cached before the error aborts the stream.
    assert mock_redis.rpush.await_count == 1


@pytest.mark.asyncio
async def test_get_streaming_skips_cache_when_error_sentinel_present():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"1")
    mock_redis.lrange = AsyncMock()

    cache = CacheManager(mock_redis, ttl=120, cache_enabled=True)
    result = await cache.get_streaming({"model": "gpt-3.5-turbo", "stream": True})

    assert result is None
    assert cache._streaming_misses == 1
    mock_redis.lrange.assert_not_called()
