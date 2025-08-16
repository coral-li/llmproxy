from unittest.mock import AsyncMock, MagicMock

import pytest

from llmproxy.clients.llm_client import LLMClient


@pytest.mark.asyncio
async def test_chat_streaming_makes_single_post():
    llm_client = LLMClient()

    try:
        mock_httpx_client = MagicMock()
        mock_httpx_client.aclose = AsyncMock()
        llm_client.client = mock_httpx_client

        # First stream context manager (used by current implementation to peek)
        first_cm = AsyncMock()
        first_resp = AsyncMock()
        first_resp.status_code = 200

        async def first_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
            yield "data: [DONE]"

        first_resp.aiter_lines = first_aiter_lines
        first_cm.__aenter__.return_value = first_resp
        first_cm.__aexit__.return_value = None

        # Second stream context manager (used by current implementation for actual streaming)
        second_cm = AsyncMock()
        second_resp = AsyncMock()
        second_resp.status_code = 200

        async def second_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
            yield "data: [DONE]"

        second_resp.aiter_lines = second_aiter_lines
        second_cm.__aenter__.return_value = second_resp
        second_cm.__aexit__.return_value = None

        # Side effect returns two different context managers (current bug path)
        mock_httpx_client.stream.side_effect = [first_cm, second_cm]

        # Invoke streaming request
        result = await llm_client.create_chat_completion(
            model="gpt-4o-mini",
            endpoint_url="https://api.openai.com",
            api_key="test-key",
            request_data={
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
            stream=True,
        )

        # Should return an async generator
        assert not isinstance(result, dict)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) > 0

        # Assert only one outbound POST (single call to httpx stream)
        # This will FAIL prior to the fix (current code makes 2 calls)
        assert mock_httpx_client.stream.call_count == 1

    finally:
        await llm_client.close()


@pytest.mark.asyncio
async def test_responses_api_streaming_makes_single_post():
    llm_client = LLMClient()

    try:
        mock_httpx_client = MagicMock()
        mock_httpx_client.aclose = AsyncMock()
        llm_client.client = mock_httpx_client

        # First stream context manager (peek)
        first_cm = AsyncMock()
        first_resp = AsyncMock()
        first_resp.status_code = 200

        async def first_aiter_lines():
            yield "event: response.output_text.delta"
            yield "data: Hello"
            yield "event: done"

        first_resp.aiter_lines = first_aiter_lines
        first_cm.__aenter__.return_value = first_resp
        first_cm.__aexit__.return_value = None

        # Second stream context manager (actual streaming)
        second_cm = AsyncMock()
        second_resp = AsyncMock()
        second_resp.status_code = 200

        async def second_aiter_lines():
            yield "event: response.output_text.delta"
            yield "data: Hello"
            yield "event: done"

        second_resp.aiter_lines = second_aiter_lines
        second_cm.__aenter__.return_value = second_resp
        second_cm.__aexit__.return_value = None

        mock_httpx_client.stream.side_effect = [first_cm, second_cm]

        # Invoke streaming request for responses API
        result = await llm_client.create_response(
            model="gpt-4o-mini",
            endpoint_url="https://api.openai.com",
            api_key="test-key",
            request_data={
                "input": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
            stream=True,
        )

        # Should return an async generator
        assert not isinstance(result, dict)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) > 0

        # Assert only one outbound POST (single call to httpx stream)
        # This will FAIL prior to the fix (current code makes 2 calls)
        assert mock_httpx_client.stream.call_count == 1

    finally:
        await llm_client.close()
