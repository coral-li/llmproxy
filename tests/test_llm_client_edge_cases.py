import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llmproxy.clients.llm_client import LLMClient


class TestLLMClientEdgeCases:
    """Edge case tests for LLMClient to uncover potential bugs"""

    @pytest.fixture
    async def client(self):
        client = LLMClient(timeout=1.0)
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_create_chat_completion_with_malformed_url(self, client):
        """Test handling of malformed URLs"""
        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.InvalidURL("Invalid URL")

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="not-a-valid-url",
                api_key="test-key",
                request_data={"messages": [{"role": "user", "content": "hi"}]},
            )

            assert result["status_code"] == 500
            assert "Invalid URL" in result["error"]

    @pytest.mark.asyncio
    async def test_create_chat_completion_with_none_request_data(self, client):
        """Test handling of None request data"""
        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_response.headers = {}
            mock_post.return_value = mock_response

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data=None,  # None request data
            )

            assert result["status_code"] == 200
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"] is None

    @pytest.mark.asyncio
    async def test_create_chat_completion_with_empty_string_endpoint(self, client):
        """Test handling of empty string endpoint"""
        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="",  # Empty endpoint
                api_key="test-key",
                request_data={"messages": []},
            )

            assert result["status_code"] == 500
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_azure_url_detection_edge_cases(self, client):
        """Test Azure URL detection with various edge cases"""
        test_cases = [
            ("https://myopenai.azure.com", True),
            ("https://myopenai.azure.com/", True),
            ("https://test.openai.azure.com", True),
            ("https://azure.com", True),
            ("https://fakeazure.com", False),
            ("https://azure.com.fake", False),
            ("azure.com", True),  # No protocol
            ("", False),  # Empty string
        ]

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": []}
            mock_response.headers = {}
            mock_post.return_value = mock_response

            for endpoint, is_azure in test_cases:
                await client.create_chat_completion(
                    model="gpt-3.5-turbo",
                    endpoint_url=endpoint,
                    api_key="test-key",
                    request_data={"messages": []},
                )

                if mock_post.called:
                    call_args = mock_post.call_args
                    url = call_args[0][0]
                    if is_azure and endpoint:
                        assert "/openai/v1/chat/completions" in url
                    elif endpoint:
                        assert "/v1/chat/completions" in url

    @pytest.mark.asyncio
    async def test_response_with_invalid_json(self, client):
        """Test handling of invalid JSON in response"""
        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "invalid json response"
            mock_response.headers = {}
            mock_post.return_value = mock_response

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={"messages": []},
            )

            # Should handle JSON decode error gracefully
            assert result["status_code"] == 500
            assert result["error"] == "invalid json response"

    @pytest.mark.asyncio
    async def test_timeout_handling_precision(self, client):
        """Test timeout handling with precise timing"""
        start_time = time.time()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={"messages": []},
            )

            duration = time.time() - start_time
            assert result["status_code"] == 504
            assert "Request timeout after 1.0s" in result["error"]
            assert result["duration_ms"] >= 0
            assert duration < 2.0  # Should fail quickly

    @pytest.mark.asyncio
    async def test_vague_400_error_logging(self, client):
        """Test special handling of vague 400 errors"""
        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request. Please check your inputs."
            mock_response.headers = {"x-request-id": "req_123"}
            mock_post.return_value = mock_response

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={
                    "messages": [{"role": "user", "content": "test"}],
                    "temperature": 1.5,  # Invalid temperature
                },
            )

            assert result["status_code"] == 400
            assert "check your inputs" in result["error"]

    @pytest.mark.asyncio
    async def test_stream_request_error_handling(self, client):
        """Test error handling in streaming requests"""
        # Create a mock that properly raises the exception during streaming
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.side_effect = httpx.ConnectError(
            "Connection failed"
        )

        with patch.object(client.client, "stream", return_value=mock_context_manager):
            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={"messages": []},
                stream=True,
            )

            assert result["status_code"] == 500
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_stream_response_malformed_sse(self, client):
        """Test handling of malformed SSE in streaming response"""
        # Test that malformed SSE data is forwarded as-is
        with patch.object(client.client, "stream") as mock_stream:
            # Create mock that returns the mock context manager when called
            mock_stream.side_effect = lambda *args, **kwargs: mock_stream

            # Set up mock as async context manager
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)

            # Mock successful response that will trigger streaming
            mock_stream.status_code = 200
            mock_stream.headers = {}

            # Set up aiter_lines to return our test data
            async def mock_aiter_lines():
                yield "data: invalid"
                yield "data: [DONE]"

            mock_stream.aiter_lines = mock_aiter_lines

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={"messages": []},
                stream=True,
            )

            # Result should be an async generator
            assert not isinstance(result, dict)

            # Collect chunks
            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            # Should forward both chunks with proper formatting
            assert len(chunks) == 2
            assert "data: invalid\n\n" in chunks
            assert "data: [DONE]\n\n" in chunks

    @pytest.mark.asyncio
    async def test_responses_api_with_complex_input(self, client):
        """Test responses API with complex input structures"""
        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"output_text": "response"}
            mock_response.headers = {}
            mock_post.return_value = mock_response

            complex_input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,abc123"},
                        },
                    ],
                }
            ]

            result = await client.create_response(
                model="gpt-4",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={"input": complex_input},
            )

            assert result["status_code"] == 200
            mock_post.assert_called_once()
            # Verify URL construction for responses API
            call_args = mock_post.call_args
            assert "/responses" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_large_request_data_handling(self, client):
        """Test handling of very large request data"""
        # Create a large message
        large_content = "x" * 100000  # 100KB message

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 413  # Payload too large
            mock_response.text = "Request entity too large"
            mock_response.headers = {}
            mock_post.return_value = mock_response

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={"messages": [{"role": "user", "content": large_content}]},
            )

            assert result["status_code"] == 413
            assert "too large" in result["error"]

    @pytest.mark.asyncio
    async def test_concurrent_requests_resource_cleanup(self, client):
        """Test that concurrent requests properly clean up resources"""

        async def slow_request():
            with patch.object(
                client.client, "post", new_callable=AsyncMock
            ) as mock_post:
                # Simulate slow response
                await asyncio.sleep(0.1)
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"choices": []}
                mock_response.headers = {}
                mock_post.return_value = mock_response

                return await client.create_chat_completion(
                    model="gpt-3.5-turbo",
                    endpoint_url="https://api.openai.com",
                    api_key="test-key",
                    request_data={"messages": []},
                )

        # Run multiple concurrent requests
        tasks = [slow_request() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            if isinstance(result, dict):
                assert "status_code" in result

    @pytest.mark.asyncio
    async def test_log_debug_details_edge_cases(self, client):
        """Test _log_debug_details with various edge cases"""
        # Test with extremely long messages
        long_messages = [
            {"role": "user", "content": "x" * 1000},  # Very long content
            {"role": "assistant", "content": None},  # None content
            {"role": "system"},  # Missing content
        ]

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "invalid_request_error: check your inputs"
            mock_response.headers = {}
            mock_post.return_value = mock_response

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={"messages": long_messages},
            )

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_filter_and_yield_chunk_edge_cases(self, client):
        """Test _filter_and_yield_chunk with various edge cases"""
        # Test with malformed JSON
        test_lines = [
            "data: {incomplete json",
            "data: null",
            "data: []",
            "data: 123",
            'data: {"choices": []}',
            "data: [DONE]",
            ": comment line",
            "",
            "invalid line format",
        ]

        for line in test_lines:
            chunks = client._filter_and_yield_chunk(line)
            # Should not raise exceptions
            assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_close_multiple_times(self, client):
        """Test that calling close() multiple times doesn't cause issues"""
        await client.close()
        await client.close()  # Second close should not raise

    @pytest.mark.asyncio
    async def test_request_with_unicode_content(self, client):
        """Test handling of Unicode content in requests"""
        unicode_content = "Hello 世界 🌍 مرحبا العالم"

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": unicode_content}}]
            }
            mock_response.headers = {}
            mock_post.return_value = mock_response

            result = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                endpoint_url="https://api.openai.com",
                api_key="test-key",
                request_data={
                    "messages": [{"role": "user", "content": unicode_content}]
                },
            )

            assert result["status_code"] == 200
            assert result["data"]["choices"][0]["message"]["content"] == unicode_content
