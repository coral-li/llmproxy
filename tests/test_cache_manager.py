import json

import redis.asyncio as redis

from llmproxy.core.cache_manager import CacheManager, StreamingCacheWriter


def test_sse_parsing_handles_data_before_event():
    """Test that SSE parsing works correctly when data lines appear before event lines"""
    cache_manager = CacheManager(
        redis.Redis(), cache_enabled=False
    )  # Mock for parsing test
    writer = StreamingCacheWriter(cache_manager, {}, is_responses_api=True)

    # Test case 1: Normal order (event first)
    chunk_normal = """event: response.created
data: {"response": {"model": "gpt-4", "created": 1234567890}}

"""
    result_normal = writer._parse_responses_event(chunk_normal)
    assert result_normal is not None
    assert result_normal.event_type == "response.created"
    assert result_normal.data_type == "response"

    # Test case 2: Data first (this was the bug)
    chunk_data_first = """data: {"response": {"model": "gpt-4", "created": 1234567890}}
event: response.created

"""
    result_data_first = writer._parse_responses_event(chunk_data_first)
    assert result_data_first is not None
    assert result_data_first.event_type == "response.created"
    assert result_data_first.data_type == "response"

    # Test case 3: Multiple data lines (edge case)
    chunk_multiple_data = """event: response.output_text.delta
data: {"delta": "Hello", "output_index": 0}
data: {"delta": " world", "output_index": 0}

"""
    result_multiple = writer._parse_responses_event(chunk_multiple_data)
    assert result_multiple is not None
    assert result_multiple.event_type == "response.output_text.delta"
    assert result_multiple.content == "Hello"  # Should use first valid data line

    # Test case 4: Invalid JSON should be skipped
    chunk_invalid_json = """event: response.created
data: invalid-json
data: {"response": {"model": "gpt-4"}}

"""
    result_invalid = writer._parse_responses_event(chunk_invalid_json)
    assert result_invalid is not None
    assert result_invalid.event_type == "response.created"


def test_parse_responses_event_preserves_ids():
    """Ensure responses API SSE parsing retains upstream identifier fields."""

    cache_manager = CacheManager(redis.Redis(), cache_enabled=False)
    writer = StreamingCacheWriter(cache_manager, {}, is_responses_api=True)

    created_chunk = (
        "event: response.created\n"
        'data: {"type": "response.created", "response": {"id": "resp_original", "model": "gpt-4", "created_at": 1700000000, "status": "in_progress"}}\n'
        "\n"
    )
    created = writer._parse_responses_event(created_chunk)
    assert created is not None
    assert created.metadata.get("response_id") == "resp_original"
    assert created.metadata.get("created_at") == 1700000000

    item_chunk = (
        "event: response.output_item.added\n"
        'data: {"type": "response.output_item.added", "response_id": "resp_original", "output_index": 0, "item": {"id": "msg_original", "type": "message", "status": "in_progress", "role": "assistant", "summary": []}}\n'
        "\n"
    )
    item = writer._parse_responses_event(item_chunk)
    assert item is not None
    assert item.metadata.get("item_id") == "msg_original"
    assert item.metadata.get("response_id") == "resp_original"

    completed_chunk = (
        "event: response.completed\n"
        'data: {"type": "response.completed", "response": {"id": "resp_original", "model": "gpt-4", "created_at": 1700000005, "status": "completed", "outputs": [{"id": "msg_original", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]}]}}\n'
        "\n"
    )
    completed = writer._parse_responses_event(completed_chunk)
    assert completed is not None
    assert completed.metadata.get("response_id") == "resp_original"
    outputs = completed.metadata.get("outputs")
    assert isinstance(outputs, list)
    assert outputs and outputs[0].get("id") == "msg_original"
    assert completed.metadata.get("created_at") == 1700000005


def test_reconstruct_responses_stream_uses_cached_ids():
    """Reconstruction should replay cached identifiers instead of minting new ones."""

    cache_manager = CacheManager(redis.Redis(), cache_enabled=False)

    normalized_chunks = [
        {
            "event_type": "response.created",
            "data_type": "response",
            "content": None,
            "metadata": {
                "model": "gpt-4",
                "created": 1700000000,
                "response_id": "resp_original",
                "status": "in_progress",
            },
        },
        {
            "event_type": "response.output_item.added",
            "data_type": "message",
            "content": None,
            "metadata": {
                "output_index": 0,
                "summary": [],
                "item_id": "msg_original",
                "item_status": "in_progress",
                "item_role": "assistant",
                "response_id": "resp_original",
            },
        },
        {
            "event_type": "response.output_text.delta",
            "data_type": "text",
            "content": "Hello ",
            "metadata": {"output_index": 0, "content_index": 0},
        },
        {
            "event_type": "response.completed",
            "data_type": "response",
            "content": None,
            "metadata": {
                "model": "gpt-4",
                "created": 1700000005,
                "response_id": "resp_original",
                "outputs": [
                    {
                        "id": "msg_original",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello "}],
                    }
                ],
            },
        },
    ]

    reconstructed = cache_manager._reconstruct_responses_stream(normalized_chunks)
    data_events = [
        json.loads(line[len("data: ") :].strip())
        for line in reconstructed
        if line.startswith("data: ")
    ]

    created_event = next(
        ev for ev in data_events if ev.get("type") == "response.created"
    )
    assert created_event["response"]["id"] == "resp_original"
    assert created_event["response"]["created_at"] == 1700000000

    output_added_event = next(
        ev for ev in data_events if ev.get("type") == "response.output_item.added"
    )
    assert output_added_event["item"]["id"] == "msg_original"

    completed_event = next(
        ev for ev in data_events if ev.get("type") == "response.completed"
    )
    assert completed_event["response"]["id"] == "resp_original"
    assert completed_event["response"]["created_at"] == 1700000005
    outputs = completed_event["response"].get("outputs", [])
    assert outputs and outputs[0].get("id") == "msg_original"


def test_completed_outputs_preserve_encrypted_content():
    """Completed outputs should preserve encrypted reasoning content for cache replay."""
    cache_manager = CacheManager(redis.Redis(), cache_enabled=False)
    writer = StreamingCacheWriter(cache_manager, {}, is_responses_api=True)

    outputs = [
        {
            "id": "rs_1",
            "type": "reasoning",
            "encrypted_content": "enc-abc",
            "content": [],
        }
    ]

    cleaned = writer._clean_completed_outputs(outputs)
    assert cleaned[0].get("encrypted_content") == "enc-abc"
