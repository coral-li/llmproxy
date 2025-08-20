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
        'data: {"type": "response.completed", "response": {"model": "gpt-4", "outputs": []}}\n'
    )
    await writer.write_and_yield("\n")  # flush & finalize

    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_non_empty_responses_api_stream_cached():
    """Ensure that a responses-API streaming result containing text IS cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}  # responses API identifier
    writer = await cache.create_streaming_cache_writer(request_data)

    # Send SSE sequence with actual text delta content
    await writer.write_and_yield("event: response.created\n")
    await writer.write_and_yield(
        'data: {"type": "response.created", "response": {"model": "gpt-4"}}\n'
    )
    await writer.write_and_yield("\n")  # flush

    await writer.write_and_yield("event: response.output_text.delta\n")
    await writer.write_and_yield(
        'data: {"type": "response.output_text.delta", "delta": "Hello, how can I help you?"}\n'
    )
    await writer.write_and_yield("\n")  # flush

    await writer.write_and_yield("event: response.completed\n")
    await writer.write_and_yield(
        'data: {"type": "response.completed", "response": {"model": "gpt-4", "outputs": [{"type": "message", "content": [{"type": "text", "text": "Hello, how can I help you?"}]}]}}\n'
    )
    await writer.write_and_yield("\n")  # flush & finalize

    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_non_streaming_responses_api_cached():
    """Ensure that a non-streaming responses-API result containing text IS cached."""

    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "input": "Test"}

    # Non-streaming responses API response format
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


@pytest.mark.asyncio
async def test_chat_completion_with_tool_calls_not_empty():
    """Test that chat completions with tool calls are NOT considered empty.

    This test reproduces a bug where responses containing tool calls (but with
    content=None) are incorrectly detected as empty and not cached.

    This test should initially FAIL, confirming the bug exists.
    """
    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4.1"}

    # Response structure based on the actual logged response that was incorrectly
    # detected as empty. The key issue is that message.content is None, but
    # message.tool_calls contains meaningful data.
    response_with_tool_calls = {
        "choices": [
            {
                "content_filter_results": {},
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "annotations": [],
                    "content": None,  # This is None, but tool_calls has data
                    "refusal": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"reasoning_steps":"1. The requested unit is \'kgCO2e/kg net weight\'.\\n2. The key components are: \'kgCO2e\' (kilograms of CO2 equivalent) per \'kg\' (kilogram) of net weight.","name":"kgCO₂e/kg","correct_spelling":null}',
                                "name": "UnitModel",
                            },
                            "id": "call_h7vv9rtj5FqGx61Yu9WmkOz9",
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1750916828,
        "id": "chatcmpl-BmZjwyYeGJ4RA1s0FCpmzU10P34VR",
        "model": "gpt-4.1-2025-04-14",
        "object": "chat.completion",
        "system_fingerprint": "fp_07e970ab25",
        "usage": {"completion_tokens": 274, "prompt_tokens": 493, "total_tokens": 767},
    }

    # This response should NOT be considered empty because it contains tool calls
    # Currently this will fail because _is_empty_response incorrectly returns True
    is_empty = cache._is_empty_response(response_with_tool_calls)
    assert not is_empty, "Response with tool calls should not be considered empty"

    # The response should be cached (not skipped)
    await cache.set(request_data, response_with_tool_calls)
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completion_with_function_call_not_empty():
    """Test that chat completions with legacy function_call are NOT considered empty.

    This tests the legacy function_call format (single function call) in addition
    to the newer tool_calls format (multiple function calls).
    """
    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-3.5-turbo"}

    # Legacy function_call format response
    response_with_function_call = {
        "choices": [
            {
                "finish_reason": "function_call",
                "index": 0,
                "message": {
                    "content": None,  # Content is None
                    "role": "assistant",
                    "function_call": {  # But function_call has data
                        "arguments": '{"location": "Boston, MA"}',
                        "name": "get_current_weather",
                    },
                },
            }
        ],
        "created": 1677652288,
        "id": "chatcmpl-123",
        "model": "gpt-3.5-turbo-0613",
        "object": "chat.completion",
        "usage": {"completion_tokens": 17, "prompt_tokens": 57, "total_tokens": 74},
    }

    # This response should NOT be considered empty because it contains a function call
    is_empty = cache._is_empty_response(response_with_function_call)
    assert not is_empty, "Response with function call should not be considered empty"

    # The response should be cached (not skipped)
    await cache.set(request_data, response_with_function_call)
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_chat_completion_with_tool_calls_not_empty():
    """Test that streaming chat completions with tool calls in delta are NOT considered empty."""
    mock_redis = AsyncMock()
    cache = CacheManager(mock_redis, ttl=300, cache_enabled=True)

    request_data = {"model": "gpt-4", "stream": True}

    # Streaming response with tool calls in delta
    streaming_response_with_tool_calls = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,  # Content is None
                    "tool_calls": [  # But delta has tool_calls
                        {
                            "index": 0,
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "New York"}',
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
        "created": 1677652288,
        "id": "chatcmpl-123",
        "model": "gpt-4-0613",
        "object": "chat.completion.chunk",
    }

    # This response should NOT be considered empty because it contains tool calls in delta
    is_empty = cache._is_empty_response(streaming_response_with_tool_calls)
    assert (
        not is_empty
    ), "Streaming response with tool calls in delta should not be considered empty"

    # The response should be cached (not skipped)
    await cache.set(request_data, streaming_response_with_tool_calls)
    mock_redis.setex.assert_called_once()
