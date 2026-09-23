"""Builders for driving a real request handler over mocked dependencies."""

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from unittest.mock import AsyncMock

from llmproxy.api.base_handler import BaseRequestHandler
from llmproxy.api.chat_completions import ChatCompletionHandler
from llmproxy.api.responses import ResponseHandler
from llmproxy.config_model import (
    CacheParams,
    GeneralSettings,
    LLMProxyConfig,
    UsageStreamParams,
)
from llmproxy.core.usage_telemetry import UsageContext, UsageRecorder
from llmproxy.models.endpoint import Endpoint


class FakeRedis:
    """Minimal stand-in capturing XADD calls, optionally failing some of them."""

    def __init__(self, fail_first: int = 0) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.fail_first = fail_first
        self.calls = 0

    async def xadd(
        self,
        stream_key: str,
        fields: Dict[str, str],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("redis is down")
        self.entries.append(
            {
                "stream_key": stream_key,
                "fields": fields,
                "maxlen": maxlen,
                "approximate": approximate,
            }
        )
        return f"{self.calls}-0"

    def records(self) -> List[dict]:
        return [json.loads(entry["fields"]["payload"]) for entry in self.entries]


def make_recorder(redis: Optional[FakeRedis] = None, **params: Any) -> tuple:
    redis = redis or FakeRedis()
    return UsageRecorder(redis, UsageStreamParams(**params)), redis


def make_context(**overrides: Any) -> UsageContext:
    defaults: Dict[str, Any] = {
        "api_surface": "responses",
        "model_group": "gpt-5",
        "streaming": False,
    }
    defaults.update(overrides)
    return UsageContext(**defaults)


def make_endpoint(host: str, model: str = "m") -> Endpoint:
    return Endpoint(model=model, weight=1, params={"base_url": f"https://{host}"})


def upstream_error(status_code: int, body: str) -> dict:
    return {"status_code": status_code, "error": body, "headers": {}, "data": None}


def general_settings(**overrides: Any) -> GeneralSettings:
    return GeneralSettings(
        bind_port=5000,
        redis_host="localhost",
        redis_port=6379,
        redis_password="",
        **overrides,
    )


def cache_config(**overrides: Any) -> CacheParams:
    return CacheParams(host="localhost", port=6379, password="", **overrides)


async def drain(stream: AsyncIterator[Any]) -> List[Any]:
    return [chunk async for chunk in stream]


@dataclass
class HandlerHarness:
    """A real handler, the mocks it runs on, and the stream it records into."""

    handler: Any
    load_balancer: AsyncMock
    cache_manager: AsyncMock
    llm_client: AsyncMock
    recorder: UsageRecorder
    redis: FakeRedis

    async def records(self) -> List[dict]:
        await self.recorder.flush()
        return self.redis.records()

    async def only_record(self) -> dict:
        (record,) = await self.records()
        return record


def make_handler(
    handler_class: Callable[..., BaseRequestHandler] = ChatCompletionHandler,
    *,
    endpoints: tuple = (),
    model_group: str = "m",
    cache: bool = False,
    retries: int = 3,
    caller_headers: tuple = (),
) -> HandlerHarness:
    """A real handler over mocked dependencies, recording into a fake stream."""
    recorder, redis = make_recorder(caller_headers=list(caller_headers))
    load_balancer = AsyncMock()
    load_balancer.get_model_groups = lambda: [model_group]
    load_balancer.endpoint_configs = {model_group: list(endpoints)}
    cache_manager = AsyncMock()
    cache_manager._should_cache = lambda _request: cache
    cache_manager.get.return_value = None
    cache_manager.get_streaming.return_value = None
    llm_client = AsyncMock()
    config = LLMProxyConfig(
        general_settings=general_settings(num_retries=retries, cache=cache),
        model_groups=[],
    )
    extra = (
        {"response_affinity_manager": AsyncMock()}
        if handler_class is ResponseHandler
        else {}
    )
    handler = handler_class(
        load_balancer=load_balancer,
        cache_manager=cache_manager,
        llm_client=llm_client,
        config=config,
        usage_recorder=recorder,
        **extra,
    )
    return HandlerHarness(
        handler, load_balancer, cache_manager, llm_client, recorder, redis
    )
