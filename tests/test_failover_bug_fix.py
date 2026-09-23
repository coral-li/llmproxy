"""
Test for the failover bug fix.

This test verifies that when endpoints fail, the retry mechanism properly
tries different endpoints instead of wasting retry attempts on the same
failed endpoint.
"""

import json
from unittest.mock import AsyncMock

import pytest
from handler_harness import (
    HandlerHarness,
    make_context,
    make_endpoint,
    make_handler,
    upstream_error,
)

from llmproxy.api.chat_completions import ChatCompletionHandler
from llmproxy.config_model import GeneralSettings, LLMProxyConfig
from llmproxy.models.endpoint import Endpoint


@pytest.mark.asyncio
async def test_failover_tries_different_endpoints():
    """Test that failover logic tries different endpoints when one fails"""

    # Create mock dependencies
    mock_load_balancer = AsyncMock()
    mock_cache_manager = AsyncMock()
    mock_llm_client = AsyncMock()

    # Create config with 3 retries
    config = LLMProxyConfig(
        general_settings=GeneralSettings(
            bind_port=5000,
            redis_host="localhost",
            redis_port=6379,
            redis_password="",
            num_retries=3,
            cache=False,
        ),
        model_groups=[],
    )

    # Create handler
    handler = ChatCompletionHandler(
        load_balancer=mock_load_balancer,
        cache_manager=mock_cache_manager,
        llm_client=mock_llm_client,
        config=config,
    )

    # Create mock endpoints
    endpoint1 = Endpoint(
        model="gpt-3.5-turbo", weight=1, params={"base_url": "https://api1.example.com"}
    )
    endpoint2 = Endpoint(
        model="gpt-3.5-turbo", weight=1, params={"base_url": "https://api2.example.com"}
    )
    endpoint3 = Endpoint(
        model="gpt-3.5-turbo", weight=1, params={"base_url": "https://api3.example.com"}
    )

    mock_load_balancer.endpoint_configs = {
        "gpt-3.5-turbo": [endpoint1, endpoint2, endpoint3]
    }

    # Mock load balancer to return different endpoints in sequence
    mock_load_balancer.select_endpoint.side_effect = [
        endpoint1,
        endpoint2,
        endpoint3,  # First round of selections
        endpoint1,
        endpoint2,
        endpoint3,  # Second round (should be skipped for attempted ones)
        endpoint1,
        endpoint2,
        endpoint3,  # Third round (should be skipped for attempted ones)
    ]
    mock_load_balancer.get_model_groups.return_value = ["gpt-3.5-turbo"]

    # Mock LLM client to always fail
    mock_llm_client.create_chat_completion.return_value = {
        "status_code": 500,
        "error": "Internal server error",
        "headers": {},
        "data": None,
    }

    # Mock cache manager
    mock_cache_manager._should_cache.return_value = False
    mock_cache_manager.get.return_value = None

    # Test request
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Execute the request (should fail after trying all endpoints)
    result = await handler._execute_with_failover(
        "gpt-3.5-turbo", request_data, make_context()
    )

    # Verify that the load balancer was called multiple times
    # It should be called more than 3 times because it tries to find different endpoints
    assert mock_load_balancer.select_endpoint.call_count >= 3

    # Verify that LLM client was called exactly 3 times (once per unique endpoint)
    assert mock_llm_client.create_chat_completion.call_count == 3

    # Verify that record_failure was called for each endpoint
    assert mock_load_balancer.record_failure.call_count == 3

    # Verify the final response indicates failure (returns last error response)
    assert result["status_code"] == 500
    assert "Internal server error" in result["error"]


@pytest.mark.asyncio
async def test_failover_stops_when_all_endpoints_attempted():
    """Test that failover stops early when all available endpoints have been attempted"""

    # Create mock dependencies
    mock_load_balancer = AsyncMock()
    mock_cache_manager = AsyncMock()
    mock_llm_client = AsyncMock()

    # Create config with many retries (more than available endpoints)
    config = LLMProxyConfig(
        general_settings=GeneralSettings(
            bind_port=5000,
            redis_host="localhost",
            redis_port=6379,
            redis_password="",
            num_retries=10,  # More retries than endpoints
            cache=False,
        ),
        model_groups=[],
    )

    # Create handler
    handler = ChatCompletionHandler(
        load_balancer=mock_load_balancer,
        cache_manager=mock_cache_manager,
        llm_client=mock_llm_client,
        config=config,
    )

    # Create only 2 mock endpoints
    endpoint1 = Endpoint(
        model="gpt-3.5-turbo", weight=1, params={"base_url": "https://api1.example.com"}
    )
    endpoint2 = Endpoint(
        model="gpt-3.5-turbo", weight=1, params={"base_url": "https://api2.example.com"}
    )

    mock_load_balancer.endpoint_configs = {"gpt-3.5-turbo": [endpoint1, endpoint2]}

    # Mock load balancer to always return the same 2 endpoints in rotation
    mock_load_balancer.select_endpoint.side_effect = [
        endpoint1,
        endpoint2,
    ] * 20  # Enough for all attempts
    mock_load_balancer.get_model_groups.return_value = ["gpt-3.5-turbo"]

    # Mock LLM client to always fail
    mock_llm_client.create_chat_completion.return_value = {
        "status_code": 500,
        "error": "Internal server error",
        "headers": {},
        "data": None,
    }

    # Mock cache manager
    mock_cache_manager._should_cache.return_value = False
    mock_cache_manager.get.return_value = None

    # Test request
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Execute the request
    result = await handler._execute_with_failover(
        "gpt-3.5-turbo", request_data, make_context()
    )

    # Should only try each endpoint once, even though num_retries=10
    assert mock_llm_client.create_chat_completion.call_count == 2
    assert mock_load_balancer.select_endpoint.call_count == 2
    assert mock_load_balancer.record_failure.call_count == 2

    # Verify the final response indicates exhausted endpoints
    assert result["status_code"] == 503
    assert "All endpoints failed" in result["error"]


@pytest.mark.asyncio
async def test_failover_returns_503_when_no_endpoints_available():
    """Test that failover returns 503 when no endpoints are available from the start"""

    # Create mock dependencies
    mock_load_balancer = AsyncMock()
    mock_cache_manager = AsyncMock()
    mock_llm_client = AsyncMock()

    # Create config
    config = LLMProxyConfig(
        general_settings=GeneralSettings(
            bind_port=5000,
            redis_host="localhost",
            redis_port=6379,
            redis_password="",
            num_retries=3,
            cache=False,
        ),
        model_groups=[],
    )

    # Create handler
    handler = ChatCompletionHandler(
        load_balancer=mock_load_balancer,
        cache_manager=mock_cache_manager,
        llm_client=mock_llm_client,
        config=config,
    )

    # Mock load balancer to return None (no endpoints available)
    mock_load_balancer.select_endpoint.return_value = None
    mock_load_balancer.get_model_groups.return_value = ["gpt-3.5-turbo"]
    mock_load_balancer.endpoint_configs = {"gpt-3.5-turbo": []}

    # Mock cache manager
    mock_cache_manager._should_cache.return_value = False
    mock_cache_manager.get.return_value = None

    # Test request
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Execute the request
    result = await handler._execute_with_failover(
        "gpt-3.5-turbo", request_data, make_context()
    )

    # Should not try any LLM client calls since no endpoints are available
    assert mock_llm_client.create_chat_completion.call_count == 0
    assert mock_load_balancer.record_failure.call_count == 0

    # Verify the final response indicates no endpoints available
    assert result["status_code"] == 503
    assert "No available endpoints" in result["error"]


@pytest.mark.asyncio
async def test_failover_succeeds_on_second_endpoint():
    """Test that failover succeeds when the second endpoint works"""

    # Create mock dependencies
    mock_load_balancer = AsyncMock()
    mock_cache_manager = AsyncMock()
    mock_llm_client = AsyncMock()

    # Create config
    config = LLMProxyConfig(
        general_settings=GeneralSettings(
            bind_port=5000,
            redis_host="localhost",
            redis_port=6379,
            redis_password="",
            num_retries=3,
            cache=False,
        ),
        model_groups=[],
    )

    # Create handler
    handler = ChatCompletionHandler(
        load_balancer=mock_load_balancer,
        cache_manager=mock_cache_manager,
        llm_client=mock_llm_client,
        config=config,
    )

    # Create mock endpoints
    endpoint1 = Endpoint(
        model="gpt-3.5-turbo", weight=1, params={"base_url": "https://api1.example.com"}
    )
    endpoint2 = Endpoint(
        model="gpt-3.5-turbo", weight=1, params={"base_url": "https://api2.example.com"}
    )

    mock_load_balancer.endpoint_configs = {"gpt-3.5-turbo": [endpoint1, endpoint2]}

    # Mock load balancer
    mock_load_balancer.select_endpoint.side_effect = [
        endpoint1,
        endpoint2,
        endpoint1,
        endpoint2,
    ]
    mock_load_balancer.get_model_groups.return_value = ["gpt-3.5-turbo"]

    # Mock LLM client: first endpoint fails, second succeeds
    mock_llm_client.create_chat_completion.side_effect = [
        {
            "status_code": 500,
            "error": "Server error",
            "headers": {},
            "data": None,
        },  # endpoint1 fails
        {
            "status_code": 200,
            "data": {"choices": [{"message": {"content": "Success!"}}]},
            "headers": {},
        },  # endpoint2 succeeds
    ]

    # Mock cache manager
    mock_cache_manager._should_cache.return_value = False
    mock_cache_manager.get.return_value = None

    # Test request
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Execute the request
    result = await handler._execute_with_failover(
        "gpt-3.5-turbo", request_data, make_context()
    )

    # Should try 2 endpoints: first fails, second succeeds
    assert mock_llm_client.create_chat_completion.call_count == 2
    assert (
        mock_load_balancer.record_failure.call_count == 1
    )  # Only first endpoint failed
    assert (
        mock_load_balancer.record_success.call_count == 1
    )  # Second endpoint succeeded

    # Verify success response
    assert result["status_code"] == 200
    assert "endpoint_base_url" in result


REQUEST = {"model": "gpt-3.5-turbo", "messages": [], "stream": False}


def _endpoint(host: str) -> Endpoint:
    return make_endpoint(host, model="gpt-3.5-turbo")


def _upstream(status_code: int, code: str = "invalid_json_schema") -> dict:
    return upstream_error(
        status_code, json.dumps({"error": {"code": code, "message": "no"}})
    )


def _handler_over(endpoints: list, upstream: dict) -> HandlerHarness:
    """A chat handler whose every upstream attempt answers `upstream`."""
    harness = make_handler(endpoints=tuple(endpoints), model_group="gpt-3.5-turbo")
    harness.load_balancer.select_endpoint.side_effect = list(endpoints)
    harness.llm_client.create_chat_completion.return_value = upstream
    return harness


async def _fail_over(harness: HandlerHarness) -> dict:
    return await harness.handler._execute_with_failover(
        "gpt-3.5-turbo", REQUEST, make_context()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 413, 422])
async def test_a_refusal_reaches_the_caller_once_every_endpoint_was_tried(
    status_code,
):
    """Not a 503, which reads as an outage and invites a retry."""
    harness = _handler_over([_endpoint("a")], _upstream(status_code))

    result = await _fail_over(harness)

    assert result["status_code"] == status_code
    assert result["error"] == _upstream(status_code)["error"]
    assert harness.llm_client.create_chat_completion.call_count == 1


@pytest.mark.asyncio
async def test_a_refusal_reaches_the_caller_when_other_endpoints_are_unavailable():
    first, second = _endpoint("a"), _endpoint("b")
    harness = _handler_over([first, second], _upstream(400))
    # The second endpoint is cooling down, so there is nothing left to try.
    harness.load_balancer.select_endpoint.side_effect = [first, None]

    result = await _fail_over(harness)

    assert result["status_code"] == 400


@pytest.mark.asyncio
async def test_every_endpoint_gets_a_chance_to_accept_a_refused_request():
    harness = _handler_over([_endpoint("a"), _endpoint("b")], _upstream(400))

    result = await _fail_over(harness)

    assert result["status_code"] == 400
    assert harness.llm_client.create_chat_completion.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, code",
    [(401, "invalid_api_key"), (403, "forbidden"), (404, "DeploymentNotFound")],
)
async def test_a_misconfigured_endpoint_is_still_the_proxys_failure(status_code, code):
    """A bad key or a missing deployment says nothing about the request."""
    harness = _handler_over([_endpoint("a")], _upstream(status_code, code))

    result = await _fail_over(harness)

    assert result["status_code"] == 503
    assert result["error_label"] == code
