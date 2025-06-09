import json
from unittest.mock import MagicMock

import pytest

from llmproxy.core.cache_manager import CacheManager, StreamingCacheWriter


@pytest.mark.parametrize("extra_body", ["string", 123, None, ["list"]])
def test_should_cache_handles_non_dict_extra_body(extra_body):
    cache = CacheManager(MagicMock(), cache_enabled=True)
    # Should not raise when extra_body is not a dict
    result = cache._should_cache({"extra_body": extra_body})
    assert result is True


def test_parse_sse_lines_handles_irregular_whitespace():
    cache = CacheManager(MagicMock(), cache_enabled=False)
    writer = StreamingCacheWriter(cache, {}, is_responses_api=True)

    chunk = 'data: {"foo": "bar"}\n event: my_event\n'
    event_type, data = writer._parse_sse_lines(chunk)
    assert event_type is None
    assert data == {"foo": "bar"}


def test_parse_sse_lines_invalid_then_valid_json():
    cache = CacheManager(MagicMock(), cache_enabled=False)
    writer = StreamingCacheWriter(cache, {}, is_responses_api=True)

    chunk = (
        "event: response.output_text.delta\n"
        "data: invalid-json\n"
        'data: {"delta": "hi"}\n'
    )
    event_type, data = writer._parse_sse_lines(chunk)
    assert event_type == "response.output_text.delta"
    assert data == {"delta": "hi"}


def test_create_response_chunk_unknown_event():
    cache = CacheManager(MagicMock(), cache_enabled=False)
    writer = StreamingCacheWriter(cache, {}, is_responses_api=True)

    chunk = writer._create_response_chunk("unknown.event", {"foo": "bar"})
    assert chunk is None


def test_detect_error_in_chunk_handles_json_error():
    cache = CacheManager(MagicMock(), cache_enabled=True)
    writer = StreamingCacheWriter(cache, {}, is_responses_api=True)

    error_chunk = 'data: {"error": "boom"}'
    assert writer._detect_error_in_chunk(error_chunk) is True
    assert writer._error_occurred is True

    writer._error_occurred = False
    ok_chunk = 'data: {"delta": "ok"}'
    assert writer._detect_error_in_chunk(ok_chunk) is False
    assert writer._error_occurred is False
