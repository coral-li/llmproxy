import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmproxy.api.responses import ResponseHandler


@pytest.mark.asyncio
async def test_cached_stream_refreshes_affinity():
    cache_manager = AsyncMock()
    cache_manager._should_cache = MagicMock(return_value=True)
    cache_manager.get_streaming = AsyncMock(
        return_value=[
            "event: response.created\n",
            'data: {"type": "response.created", "response": {"id": "resp_cached"}}\n\n',
        ]
    )
    cache_manager.get_affinity = AsyncMock(return_value="endpoint-123")

    response_affinity_manager = AsyncMock()
    handler = ResponseHandler(
        load_balancer=MagicMock(),
        cache_manager=cache_manager,
        llm_client=MagicMock(),
        config=SimpleNamespace(general_settings=SimpleNamespace(cache=True)),
        response_affinity_manager=response_affinity_manager,
    )

    response = await handler._check_cache(
        {"model": "gpt-3.5-turbo", "stream": True},
        True,
        time.time(),
        "gpt-3.5-turbo",
    )

    assert response is not None
    assert response.headers.get("X-Proxy-Cache-Hit") == "true"

    async for _ in response.body_iterator:
        pass

    assert response_affinity_manager.set_endpoint_id.await_count >= 1
    response_affinity_manager.set_endpoint_id.assert_any_await(
        "resp_cached", "endpoint-123"
    )
