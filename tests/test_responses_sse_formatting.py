import json
from unittest.mock import MagicMock

from llmproxy.api.responses import ResponseHandler


def test_response_created_preserves_sse_terminator_when_rewriting():
    handler = ResponseHandler(
        load_balancer=MagicMock(),
        cache_manager=MagicMock(),
        llm_client=MagicMock(),
        config=MagicMock(),
        response_affinity_manager=MagicMock(),
    )

    payload = {
        "type": "response.created",
        "response": {"id": "resp_123", "created": 1712345678},
    }
    line = f"data: {json.dumps(payload)}\n\n"

    updated = handler._ensure_created_at_in_stream_line(line, "response.created")

    assert updated.endswith("\n\n")
    updated_payload = json.loads(updated[len("data: ") :].strip())
    assert updated_payload["response"]["created_at"] == 1712345678
