import asyncio
from types import SimpleNamespace
from typing import AsyncIterator, Callable, List, Optional

import pytest
from fastapi import HTTPException, Request

from llmproxy.api.routes import _handle_endpoint, _read_limited_json
from llmproxy.core.cache_manager import CacheManager


class InMemoryRedis:
    def __init__(self):
        self.values = {}
        self.lists = {}
        self.expirations = {}

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.values[key] = value
        self.expirations[key] = ttl
        return True

    async def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    async def expire(self, key, ttl):
        self.expirations[key] = ttl
        return key in self.values or key in self.lists

    async def delete(self, *keys):
        deleted = 0
        for key in keys:
            deleted += int(self.values.pop(key, None) is not None)
            deleted += int(self.lists.pop(key, None) is not None)
            self.expirations.pop(key, None)
        return deleted

    async def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    async def rename(self, source, destination):
        if source in self.lists:
            self.lists[destination] = self.lists.pop(source)
        elif source in self.values:
            self.values[destination] = self.values.pop(source)
        else:
            raise KeyError(source)

        if source in self.expirations:
            self.expirations[destination] = self.expirations.pop(source)
        return True


def build_request(
    chunks: List[bytes],
    *,
    content_length: Optional[int] = None,
) -> Request:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat/completions",
        "headers": headers,
    }
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]
    messages.append({"type": "http.request", "body": b"", "more_body": False})
    message_iter = iter(messages)

    async def receive():
        return next(message_iter)

    return Request(scope, receive)


def config_provider(limit: int) -> Callable[[], SimpleNamespace]:
    return lambda: SimpleNamespace(
        general_settings=SimpleNamespace(max_request_body_bytes=limit)
    )


@pytest.mark.asyncio
async def test_read_limited_json_rejects_oversized_content_length():
    request = build_request([b'{"model": "gpt-4"}'], content_length=1024)

    with pytest.raises(HTTPException) as exc_info:
        await _read_limited_json(request, max_body_bytes=16)

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_read_limited_json_rejects_chunked_body_over_limit():
    request = build_request([b'{"prompt":"', b"x" * 32, b'"}'])

    with pytest.raises(HTTPException) as exc_info:
        await _read_limited_json(request, max_body_bytes=16)

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_handle_endpoint_allows_under_limit_json_to_reach_handler():
    request = build_request([b'{"model": "gpt-4"}'])
    observed_request_data = None

    async def process_func(handler, request_data):
        nonlocal observed_request_data
        observed_request_data = request_data
        return {"ok": True}

    response = await _handle_endpoint(
        request,
        get_handler=lambda: object(),
        process_func=process_func,
        config_provider=config_provider(1024),
    )

    assert response == {"ok": True}
    assert observed_request_data == {"model": "gpt-4"}


@pytest.mark.asyncio
async def test_oversized_non_streaming_response_is_not_cached():
    redis = InMemoryRedis()
    cache = CacheManager(redis, cache_enabled=True, max_cache_entry_bytes=64)
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    response = {"choices": [{"message": {"content": "x" * 128}}]}

    await cache.set(request, response)

    assert redis.values == {}


@pytest.mark.asyncio
async def test_under_limit_non_streaming_response_is_cached():
    redis = InMemoryRedis()
    cache = CacheManager(redis, cache_enabled=True, max_cache_entry_bytes=1024)
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    response = {"choices": [{"message": {"content": "ok"}}]}

    await cache.set(request, response)

    assert await cache.get(request) == response


@pytest.mark.asyncio
async def test_chat_stream_over_cache_limit_yields_but_does_not_cache():
    redis = InMemoryRedis()
    cache = CacheManager(redis, cache_enabled=True, max_cache_entry_bytes=64)
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    writer = await cache.create_streaming_cache_writer(request)
    chunks = [
        f'data: {{"choices": [{{"delta": {{"content": "{("x" * 128)}"}}}}]}}',
        "data: [DONE]",
    ]

    async def upstream() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    collected = []
    async for chunk in writer.intercept_stream(upstream()):
        collected.append(chunk)

    assert collected == chunks
    assert await cache.get_streaming(request) is None


@pytest.mark.asyncio
async def test_responses_stream_over_cache_limit_yields_but_does_not_cache():
    redis = InMemoryRedis()
    cache = CacheManager(redis, cache_enabled=True, max_cache_entry_bytes=128)
    request = {"model": "gpt-4", "input": "hi"}
    writer = await cache.create_streaming_cache_writer(request)
    chunks = [
        "event: response.created\n",
        'data: {"type": "response.created", "response": {"model": "gpt-4"}}\n',
        "\n",
        "event: response.output_text.delta\n",
        f'data: {{"type": "response.output_text.delta", "delta": "{("x" * 256)}"}}\n',
        "\n",
        "event: response.completed\n",
        'data: {"type": "response.completed", "response": {"model": "gpt-4", "outputs": [{"type": "message", "content": [{"type": "text", "text": "done"}]}]}}\n',
        "\n",
    ]

    async def upstream() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)

    collected = []
    async for chunk in writer.intercept_stream(upstream()):
        collected.append(chunk)

    assert collected == chunks
    assert await cache.get_streaming(request) is None
