import time

import pytest


@pytest.mark.usefixtures("clear_cache")
def test_live_failover_retry_with_openai_client(openai_client, llmproxy_server, model):
    """End-to-end: first two upstreams return retryable errors, proxy retries and succeeds on fallback.

    Setup:
    - Configure mock servers on ports 8001 and 8002 to return 500.
    - Leave 8003 as healthy.
    - Use test config with all weights 0 so primary selection is arbitrary but
      we only need that the first attempt can land on an error endpoint.
    Expectation:
    - The OpenAI client call succeeds because llmproxy retries on another endpoint.
    - The returned text contains the port tag of the healthy endpoint (8003).
    """

    # Arrange: force errors on the first two upstreams
    assert len(llmproxy_server.mock_servers) >= 3, "Expected 3 mock servers"
    llmproxy_server.mock_servers[0].set_error_response(500)
    llmproxy_server.mock_servers[1].set_error_response(500)
    llmproxy_server.mock_servers[2].clear_error_response()

    # Act: make a real request via OpenAI client to llmproxy
    start = time.time()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Say 'Hello from LLMProxy!' and nothing else.",
            }
        ],
        extra_body={"cache": {"no-cache": True}},
    )
    duration = time.time() - start

    # Assert: request succeeded and was served by healthy endpoint (tagged by mock)
    assert response.choices and response.choices[0].message.content
    content = response.choices[0].message.content
    # Should contain our success phrase and the mock port tag of 8003
    assert "Hello from LLMProxy!" in content
    assert "[from 8003]" in content, f"Unexpected origin: {content}"
    # Sanity check duration > 0
    assert duration >= 0
