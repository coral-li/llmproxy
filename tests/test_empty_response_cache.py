from unittest.mock import MagicMock

import pytest

from llmproxy.core.cache_manager import CacheManager


@pytest.mark.asyncio
async def test_empty_chat_completion_not_cached():
    """Ensure that an empty chat completion response is NOT cached."""

    mock_redis = MagicMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo"}
    # Simulate a response with empty content
    empty_response = {
        "id": "chatcmpl-empty",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
    }

    # Attempt to cache the empty response
    await cache.set(request_data, empty_response)

    # Redis should not be invoked to store the value
    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_non_empty_chat_completion_is_cached():
    """Ensure that a non-empty chat completion IS cached (control check)."""

    mock_redis = MagicMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo"}
    valid_response = {
        "id": "chatcmpl-valid",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
    }

    await cache.set(request_data, valid_response)

    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_empty_responses_api_stream_not_cached():
    """Ensure that an empty responses-API streaming result is NOT cached."""

    mock_redis = MagicMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}  # responses API identifier
    writer = await cache.create_streaming_cache_writer(request_data)

    # Send minimal SSE sequence without any text delta content
    await writer.write_and_yield("event: response.created\n")
    await writer.write_and_yield(
        'data: {"type": "response.created", "response": {"model": "gpt-4"}}\n'
    )
    await writer.write_and_yield("\n")  # flush

    await writer.write_and_yield("event: response.completed\n")
    await writer.write_and_yield(
        'data: {"type": "response.completed", "response": {"model": "gpt-4"}}\n'
    )
    await writer.write_and_yield("\n")  # flush & finalize

    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_non_empty_responses_api_stream_cached():
    """Ensure that a responses-API streaming result containing text IS cached."""

    mock_redis = MagicMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}
    writer = await cache.create_streaming_cache_writer(request_data)

    # created event
    await writer.write_and_yield("event: response.created\n")
    await writer.write_and_yield(
        'data: {"type": "response.created", "response": {"model": "gpt-4"}}\n'
    )
    await writer.write_and_yield("\n")

    # include one text delta with content
    await writer.write_and_yield("event: response.output_text.delta\n")
    await writer.write_and_yield(
        'data: {"type": "response.output_text.delta", "delta": "Hello"}\n'
    )
    await writer.write_and_yield("\n")

    # completion
    await writer.write_and_yield("event: response.completed\n")
    await writer.write_and_yield(
        'data: {"type": "response.completed", "response": {"model": "gpt-4"}}\n'
    )
    await writer.write_and_yield("\n")

    # setex should have been called once to store normalized chunks
    mock_redis.setex.assert_called_once()
