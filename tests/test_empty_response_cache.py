from unittest.mock import AsyncMock

import pytest

from llmproxy.core.cache_manager import CacheManager


@pytest.mark.asyncio
async def test_empty_chat_completion_not_cached():
    """Ensure that an empty chat completion response is NOT cached."""

    mock_redis = AsyncMock()
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

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo"}
    # Use a realistic streaming response format that has content in delta
    valid_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    }

    await cache.set(request_data, valid_response)

    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_empty_responses_api_stream_not_cached():
    """Ensure that an empty responses-API streaming result is NOT cached."""

    mock_redis = AsyncMock()
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

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}

    # Test non-streaming responses API response format (this should be cached but currently isn't due to bug)
    responses_api_response = {
        "id": "resp_123",
        "object": "response",
        "created": 1677652288,
        "model": "gpt-4",
        "outputs": [
            {
                "type": "message",
                "content": [{"type": "text", "text": "Hello, how can I help you?"}],
            }
        ],
    }

    await cache.set(request_data, responses_api_response)
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_regular_chat_completion_with_content_cached():
    """Test that regular (non-streaming) chat completions with content are cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Regular chat completion format (non-streaming)
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_empty_streaming_chat_completion_not_cached():
    """Test that streaming chat completions with empty delta content are not cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo", "stream": True}

    # Streaming response with empty delta content
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "delta": {"content": ""},  # Empty content
                "finish_reason": None,
            }
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_responses_api_empty_outputs_not_cached():
    """Test that responses API with empty outputs is not cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}

    # Response with empty outputs array
    response = {
        "id": "resp_123",
        "object": "response",
        "created": 1677652288,
        "model": "gpt-4",
        "outputs": [],  # Empty outputs
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_responses_api_empty_text_content_not_cached():
    """Test that responses API with empty text content is not cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}

    # Response with empty text content
    response = {
        "id": "resp_123",
        "object": "response",
        "created": 1677652288,
        "model": "gpt-4",
        "outputs": [
            {
                "type": "message",
                "content": [{"type": "text", "text": ""}],  # Empty text
            }
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_responses_api_whitespace_only_not_cached():
    """Test that responses API with whitespace-only content is not cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}

    # Response with whitespace-only content
    response = {
        "id": "resp_123",
        "object": "response",
        "created": 1677652288,
        "model": "gpt-4",
        "outputs": [
            {
                "type": "message",
                "content": [{"type": "text", "text": "   \n\t  "}],  # Only whitespace
            }
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_responses_api_multiple_outputs_with_content_cached():
    """Test that responses API with multiple outputs containing content is cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}

    # Response with multiple outputs, one has content
    response = {
        "id": "resp_123",
        "object": "response",
        "created": 1677652288,
        "model": "gpt-4",
        "outputs": [
            {
                "type": "message",
                "content": [{"type": "text", "text": ""}],  # Empty
            },
            {
                "type": "message",
                "content": [{"type": "text", "text": "I can help you!"}],  # Has content
            },
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_multiple_choice_chat_completion_mixed_content():
    """Test chat completion with multiple choices where some have content."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo", "n": 2}

    # Multiple choices, one empty, one with content
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},  # Empty
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!",
                },  # Has content
                "finish_reason": "stop",
            },
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_legacy_completions_format_with_text():
    """Test legacy completions format using 'text' field."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo-instruct", "prompt": "Hello"}

    # Legacy completions format with text field
    response = {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {
                "index": 0,
                "text": "Hello! How can I assist you today?",  # Uses 'text' not 'message'
                "finish_reason": "stop",
            }
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_malformed_response_structure_handled():
    """Test that malformed response structures don't cause errors."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo"}

    # Malformed response - should not cache but not crash
    response = {
        "choices": [
            {
                "message": "not a dict",  # Invalid structure
            }
        ],
    }

    await cache.set(request_data, response)
    mock_redis.setex.assert_not_called()  # Should not cache malformed responses
