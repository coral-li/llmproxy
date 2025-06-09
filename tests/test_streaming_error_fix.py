"""
Test for the streaming error status code bug fix.

This test verifies that when a streaming request fails (e.g., 401, 429, 500),
the proxy returns the correct error status code instead of always returning 200.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llmproxy.api.chat_completions import ChatCompletionHandler
from llmproxy.api.responses import ResponseHandler
from llmproxy.clients.llm_client import LLMClient
from llmproxy.models.endpoint import Endpoint


@pytest.mark.asyncio
async def test_chat_completion_streaming_error_returns_correct_status():
    """Test that streaming errors return correct status codes for chat completions"""
    # Mock the httpx client to simulate a 401 error
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()  # Mock the aclose method
    mock_response = AsyncMock()
    mock_response.status_code = 401
    mock_response.headers = {"x-request-id": "test-123"}
    mock_response.aread = AsyncMock(
        return_value=b'{"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}}'
    )

    # Mock the stream context manager
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_response
    mock_stream_cm.__aexit__.return_value = None
    mock_client.stream.return_value = mock_stream_cm

    # Create LLM client with mocked httpx client
    llm_client = LLMClient()
    llm_client.client = mock_client

    # Create a handler (we'll mock the dependencies)
    handler = ChatCompletionHandler(
        load_balancer=MagicMock(),
        cache_manager=MagicMock(),
        llm_client=llm_client,
        config=MagicMock(),
    )

    # Create test endpoint
    endpoint = Endpoint(
        model="gpt-3.5-turbo",
        weight=1,
        params={"api_key": "invalid-key", "base_url": "https://api.openai.com"},
    )

    # Make streaming request
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Call _make_request which should now return an error dict
    result = await handler._make_request(endpoint, request_data, is_streaming=True)

    # Verify the result is an error dict with correct status code
    assert isinstance(result, dict)
    assert result["status_code"] == 401
    assert result["data"] is None
    assert "Invalid API key" in result["error"]

    await llm_client.close()


@pytest.mark.asyncio
async def test_responses_api_streaming_error_returns_correct_status():
    """Test that streaming errors return correct status codes for responses API"""
    # Mock the httpx client to simulate a 429 rate limit error
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()  # Mock the aclose method
    mock_response = AsyncMock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "60"}
    mock_response.aread = AsyncMock(
        return_value=b'{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}'
    )

    # Mock the stream context manager
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_response
    mock_stream_cm.__aexit__.return_value = None
    mock_client.stream.return_value = mock_stream_cm

    # Create LLM client with mocked httpx client
    llm_client = LLMClient()
    llm_client.client = mock_client

    # Create a handler
    handler = ResponseHandler(
        load_balancer=MagicMock(),
        cache_manager=MagicMock(),
        llm_client=llm_client,
        config=MagicMock(),
    )

    # Create test endpoint
    endpoint = Endpoint(
        model="gpt-4",
        weight=1,
        params={"api_key": "test-key", "base_url": "https://api.openai.com"},
    )

    # Make streaming request
    request_data = {
        "model": "gpt-4",
        "input": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Call _make_request which should now return an error dict
    result = await handler._make_request(endpoint, request_data, is_streaming=True)

    # Verify the result is an error dict with correct status code
    assert isinstance(result, dict)
    assert result["status_code"] == 429
    assert result["data"] is None
    assert "Rate limit exceeded" in result["error"]

    await llm_client.close()


@pytest.mark.asyncio
async def test_streaming_timeout_returns_504():
    """Test that streaming timeouts return 504 status code"""
    # Mock the httpx client to simulate a timeout
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()  # Mock the aclose method
    mock_client.stream.side_effect = httpx.TimeoutException("Request timed out")

    # Create LLM client with mocked httpx client
    llm_client = LLMClient(timeout=5.0)
    llm_client.client = mock_client

    # Test chat completion streaming timeout
    response = await llm_client.create_chat_completion(
        model="gpt-3.5-turbo",
        endpoint_url="https://api.openai.com",
        api_key="test-key",
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
        stream=True,
    )

    # Should return error dict with 504 status
    assert isinstance(response, dict)
    assert response["status_code"] == 504
    assert "timeout" in response["error"].lower()

    await llm_client.close()


@pytest.mark.asyncio
async def test_streaming_success_returns_generator():
    """Test that successful streaming returns an async generator"""
    # Mock successful streaming response
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()  # Mock the aclose method
    mock_response = AsyncMock()
    mock_response.status_code = 200

    # Mock aiter_lines to return some chunks
    async def mock_aiter_lines():
        yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
        yield "data: [DONE]"

    mock_response.aiter_lines = mock_aiter_lines

    # Mock the stream context manager
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_response
    mock_stream_cm.__aexit__.return_value = None
    mock_client.stream.return_value = mock_stream_cm

    # Create LLM client with mocked httpx client
    llm_client = LLMClient()
    llm_client.client = mock_client

    # Make streaming request
    response = await llm_client.create_chat_completion(
        model="gpt-3.5-turbo",
        endpoint_url="https://api.openai.com",
        api_key="test-key",
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
        stream=True,
    )

    # Should return an async generator, not a dict
    assert not isinstance(response, dict)

    # Verify it's an async generator by checking if we can iterate
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0
    assert any("Hello" in chunk for chunk in chunks)

    await llm_client.close()
