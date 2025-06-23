from unittest.mock import AsyncMock, MagicMock

import pytest

from llmproxy.api.embeddings import EmbeddingHandler
from llmproxy.config.config_loader import load_config
from llmproxy.core.cache_manager import CacheManager
from llmproxy.models.endpoint import Endpoint


@pytest.fixture
def mock_config():
    return load_config("test-config.yaml")


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


@pytest.fixture
def mock_load_balancer():
    lb = MagicMock()
    mock_endpoint = Endpoint(
        model="text-embedding-3-small",
        weight=1,
        params={"api_key": "test_key", "base_url": "https://api.openai.com"},
    )
    lb.select_endpoint = AsyncMock(return_value=mock_endpoint)
    lb.get_model_groups = MagicMock(return_value=["text-embedding-3-small"])
    lb.record_success = AsyncMock()
    lb.record_failure = AsyncMock()
    return lb


@pytest.fixture
def mock_cache_manager():
    return AsyncMock(spec=CacheManager)


@pytest.fixture
def embedding_handler(
    mock_load_balancer, mock_cache_manager, mock_llm_client, mock_config
):
    return EmbeddingHandler(
        load_balancer=mock_load_balancer,
        cache_manager=mock_cache_manager,
        llm_client=mock_llm_client,
        config=mock_config,
    )


@pytest.mark.asyncio
async def test_embedding_handler_happy_path(
    embedding_handler, mock_cache_manager, mock_llm_client
):
    """Test happy-path forwarding and correct response format."""
    mock_cache_manager.get.return_value = None
    request_data = {
        "model": "text-embedding-3-small",
        "input": "The food was delicious and the waiter...",
    }
    mock_llm_client.create_embedding.return_value = {
        "status_code": 200,
        "data": {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-3-small-001",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    }

    response = await embedding_handler.handle_request(request_data)

    assert "_proxy_cache_hit" in response
    assert not response["_proxy_cache_hit"]
    assert "data" in response
    assert response["data"][0]["object"] == "embedding"
    mock_llm_client.create_embedding.assert_called_once()


@pytest.mark.asyncio
async def test_embedding_cache(embedding_handler, mock_cache_manager):
    """Test that identical requests hit the cache on the second call."""
    request_data = {
        "model": "text-embedding-3-small",
        "input": "This is a test input.",
    }
    cached_response = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.3, 0.4], "index": 0}],
    }

    # First call - cache miss
    mock_cache_manager.get.return_value = None
    embedding_handler.llm_client.create_embedding.return_value = {
        "status_code": 200,
        "data": cached_response,
    }
    await embedding_handler.handle_request(request_data)

    # Second call - cache hit
    mock_cache_manager.get.return_value = cached_response
    response = await embedding_handler.handle_request(request_data)

    assert response["_proxy_cache_hit"]
    assert embedding_handler.llm_client.create_embedding.call_count == 1
    assert mock_cache_manager.get.call_count == 2


@pytest.mark.asyncio
async def test_embedding_response_not_considered_empty():
    """Test that embeddings responses are not considered empty by the cache manager."""
    from llmproxy.core.cache_manager import CacheManager

    # Create a real cache manager (not mocked) to test the actual logic
    mock_redis = AsyncMock()
    cache_manager = CacheManager(mock_redis, cache_enabled=True)

    # Test embeddings response format
    embeddings_response = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }

    # This should NOT be considered empty
    is_empty = cache_manager._is_empty_response(embeddings_response)
    assert not is_empty, "Embeddings response should not be considered empty"


@pytest.mark.asyncio
async def test_embedding_failover(
    embedding_handler, mock_load_balancer, mock_llm_client
):
    """Test that the handler retries on a secondary endpoint if the primary fails."""
    embedding_handler.cache_manager.get.return_value = None
    request_data = {
        "model": "text-embedding-3-small",
        "input": "Test failover scenario.",
    }

    endpoint1 = Endpoint(
        model="text-embedding-3-small",
        weight=1,
        params={"api_key": "key1", "base_url": "https://api.openai.com/v1"},
    )
    endpoint2 = Endpoint(
        model="text-embedding-3-small",
        weight=1,
        params={"api_key": "key2", "base_url": "https://api.azure.com/v1"},
    )

    mock_load_balancer.select_endpoint.side_effect = [endpoint1, endpoint2]

    mock_llm_client.create_embedding.side_effect = [
        {"status_code": 500, "error": "Internal Server Error"},
        {
            "status_code": 200,
            "data": {"object": "list", "data": [], "model": "text-embedding-3-small"},
        },
    ]

    response = await embedding_handler.handle_request(request_data)

    assert response is not None
    assert mock_load_balancer.select_endpoint.call_count == 2
    assert mock_llm_client.create_embedding.call_count == 2
    mock_load_balancer.record_failure.assert_called_once_with(
        endpoint1, "Internal Server Error"
    )
    mock_load_balancer.record_success.assert_called_once_with(endpoint2)
