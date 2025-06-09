import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmproxy.api.base_handler import BaseRequestHandler
from llmproxy.models.endpoint import Endpoint
from llmproxy.config_model import GeneralSettings, LLMProxyConfig


class DummyHandler(BaseRequestHandler):
    async def _make_request(self, endpoint: Endpoint, request_data: dict, is_streaming: bool) -> dict:
        return {"status_code": 200, "data": {}}


def create_handler(global_cache: bool = True, should_cache: bool = True, load_balancer=None) -> DummyHandler:
    if load_balancer is None:
        load_balancer = AsyncMock()
        load_balancer.get_model_groups.return_value = []
    cache_manager = MagicMock()
    cache_manager._should_cache.return_value = should_cache
    config = LLMProxyConfig(
        general_settings=GeneralSettings(
            bind_port=8000,
            redis_host="localhost",
            redis_port=6379,
            redis_password="",
            cache=global_cache,
        ),
        model_groups=[],
    )
    return DummyHandler(load_balancer, cache_manager, MagicMock(), config)


def test_filter_proxy_params_various_inputs():
    handler = create_handler()

    data = {
        "model": "gpt-4",
        "cache": {"no-cache": True},
        "extra_body": {"cache": {"no-cache": True}, "foo": "bar"},
        "number": 1,
    }
    result = handler._filter_proxy_params(data)
    assert result == {"model": "gpt-4", "extra_body": {"foo": "bar"}, "number": 1}

    data2 = {"cache": False, "extra_body": ["x", {"cache": True}]}
    result2 = handler._filter_proxy_params(data2)
    assert result2 == {"extra_body": ["x", {"cache": True}]}

    data3 = {"extra_body": {"cache": True}}
    assert handler._filter_proxy_params(data3) == {}


def test_is_cache_enabled_combinations():
    assert create_handler(global_cache=False, should_cache=True)._is_cache_enabled({}) is False
    assert create_handler(global_cache=True, should_cache=False)._is_cache_enabled({}) is False
    assert create_handler(global_cache=True, should_cache=True)._is_cache_enabled({}) is True


@pytest.mark.asyncio
async def test_handle_exception_streaming_and_non_streaming():
    lb = AsyncMock()
    handler = create_handler(load_balancer=lb)
    endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params={"api_key": "k"})

    # Streaming case
    result = await handler._handle_exception(endpoint, Exception("boom\n✨"), True)
    assert result["status_code"] == 500
    stream = result["data"]
    chunks = []
    async for c in stream:
        chunks.append(c.decode())
    assert chunks[-1] == "data: [DONE]\n\n"
    lb.record_failure.assert_awaited_with(endpoint, "boom\n✨")

    # Non-streaming case
    lb.reset_mock()
    result2 = await handler._handle_exception(endpoint, RuntimeError("oops"), False)
    assert result2 == {
        "status_code": 500,
        "headers": {},
        "data": None,
        "error": "oops",
    }
    lb.record_failure.assert_awaited_with(endpoint, "oops")
