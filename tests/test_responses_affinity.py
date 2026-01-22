import hashlib
import re

import httpx
import redis

from llmproxy.models.endpoint import Endpoint


def _extract_output_text(response_json: dict) -> str:
    outputs = response_json.get("output", [])
    if not isinstance(outputs, list):
        return ""
    for item in outputs:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    return text
    return ""


def _extract_port(output_text: str) -> str:
    match = re.search(r"\[from (\d+)\]", output_text)
    assert match, f"Could not find port marker in output: {output_text}"
    return match.group(1)


def test_encrypted_reasoning_without_mapping_returns_conflict(proxy_url, model):
    payload = {
        "model": model,
        "input": [{"type": "reasoning", "encrypted_content": "abc123"}],
    }

    response = httpx.post(f"{proxy_url}/responses", json=payload, timeout=10.0)

    assert response.status_code == 409
    detail = response.json().get("detail", "")
    assert "Unknown encrypted reasoning content" in detail


def test_previous_response_id_affinity(proxy_url, model):
    payload_1 = {
        "model": model,
        "input": "Hello there",
        "extra_body": {"cache": {"no-cache": True}},
    }

    response_1 = httpx.post(f"{proxy_url}/responses", json=payload_1, timeout=10.0)
    assert response_1.status_code == 200
    data_1 = response_1.json()
    response_id = data_1.get("id")
    assert response_id, "Missing response id"
    output_text_1 = _extract_output_text(data_1)
    port_1 = _extract_port(output_text_1)

    payload_2 = {
        "model": model,
        "input": "Follow up",
        "previous_response_id": response_id,
        "extra_body": {"cache": {"no-cache": True}},
    }

    response_2 = httpx.post(f"{proxy_url}/responses", json=payload_2, timeout=10.0)
    assert response_2.status_code == 200
    data_2 = response_2.json()
    output_text_2 = _extract_output_text(data_2)
    port_2 = _extract_port(output_text_2)

    assert port_1 == port_2


def test_encrypted_reasoning_affinity_mapping(proxy_url, model, llmproxy_server):
    encrypted_content = "enc-test-1"
    for mock_server in llmproxy_server.mock_servers:
        mock_server.clear_error_response()
    endpoint = Endpoint(
        model=model,
        weight=1,
        params={"api_key": "test-key-1", "base_url": "http://localhost:8001"},
    )
    encrypted_hash = hashlib.sha256(encrypted_content.encode("utf-8")).hexdigest()
    key = f"llmproxy:response_affinity:encrypted:{encrypted_hash}"

    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.set(key, endpoint.id)

    payload = {
        "model": model,
        "input": [
            {"type": "reasoning", "encrypted_content": encrypted_content},
            {"role": "user", "content": "Hello there"},
        ],
        "extra_body": {"cache": {"no-cache": True}},
    }

    response = httpx.post(f"{proxy_url}/responses", json=payload, timeout=10.0)
    assert response.status_code == 200
    output_text = _extract_output_text(response.json())
    port = _extract_port(output_text)
    assert port == "8001"
