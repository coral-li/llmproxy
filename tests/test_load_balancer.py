"""Tests for the load balancer module"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llmproxy.config_model import (
    GeneralSettings,
    LLMProxyConfig,
    ModelConfig,
    ModelGroup,
)
from llmproxy.managers.endpoint_state_manager import EndpointStateManager
from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.models.endpoint import Endpoint


class TestLoadBalancer:
    """Test cases for LoadBalancer class"""

    def test_load_balancer_initialization(self):
        """Test LoadBalancer initialization"""
        lb = LoadBalancer(cooldown_time=120, allowed_fails=2)

        assert lb.cooldown_time == 120
        assert lb.allowed_fails == 2
        assert lb.endpoint_configs == {}
        assert lb.state_manager is None

    def test_load_balancer_default_initialization(self):
        """Test LoadBalancer initialization with defaults"""
        lb = LoadBalancer()

        assert lb.cooldown_time == 60
        assert lb.allowed_fails == 1
        assert lb.endpoint_configs == {}

    def test_set_state_manager(self):
        """Test setting state manager"""
        lb = LoadBalancer()
        mock_state_manager = Mock(spec=EndpointStateManager)

        lb.set_state_manager(mock_state_manager)

        assert lb.state_manager == mock_state_manager

    @pytest.mark.asyncio
    async def test_initialize_from_config(self):
        """Test initializing from config"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)
        lb = LoadBalancer(state_manager=mock_state_manager)

        # Create test config
        config = LLMProxyConfig(
            general_settings=GeneralSettings(
                bind_port=8000,
                redis_host="localhost",
                redis_port=6379,
                redis_password="",
                allowed_fails=2,
            ),
            model_groups=[
                ModelGroup(
                    model_group="gpt-3.5-turbo",
                    models=[
                        ModelConfig(
                            model="gpt-3.5-turbo",
                            weight=1,
                            params={
                                "api_key": "test-key",
                                "base_url": "https://api.openai.com",
                            },
                        ),
                        ModelConfig(
                            model="gpt-3.5-turbo",
                            weight=0,
                            params={
                                "api_key": "backup-key",
                                "base_url": "https://backup.api.com",
                            },
                        ),
                    ],
                )
            ],
        )

        await lb.initialize_from_config(config)

        # Verify endpoints were configured
        assert "gpt-3.5-turbo" in lb.endpoint_configs
        assert len(lb.endpoint_configs["gpt-3.5-turbo"]) == 2

        # Verify state manager was called for each endpoint
        assert mock_state_manager.initialize_endpoint.call_count == 2

    @pytest.mark.asyncio
    async def test_initialize_from_config_without_state_manager(self):
        """Test initializing from config without state manager"""
        lb = LoadBalancer()

        config = LLMProxyConfig(
            general_settings=GeneralSettings(
                bind_port=8000,
                redis_host="localhost",
                redis_port=6379,
                redis_password="",
                allowed_fails=1,
            ),
            model_groups=[
                ModelGroup(
                    model_group="test-model",
                    models=[
                        ModelConfig(
                            model="test-model", weight=1, params={"api_key": "test"}
                        )
                    ],
                )
            ],
        )

        await lb.initialize_from_config(config)

        # Should still configure endpoints even without state manager
        assert "test-model" in lb.endpoint_configs
        assert len(lb.endpoint_configs["test-model"]) == 1

    @pytest.mark.asyncio
    async def test_select_endpoint_no_endpoints_configured(self):
        """Test selecting endpoint when none are configured"""
        lb = LoadBalancer()

        result = await lb.select_endpoint("nonexistent-model")

        assert result is None

    @pytest.mark.asyncio
    async def test_select_endpoint_with_primary_available(self):
        """Test selecting endpoint when primary endpoints are available"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)
        mock_state_manager.get_availability_bulk.return_value = (
            {"test-model": True, "test-model-2": True},
            {"test-model": {}, "test-model-2": {}},
        )

        lb = LoadBalancer(state_manager=mock_state_manager)

        # Create test endpoints
        endpoint1 = Endpoint(model="test", weight=2, params={"api_key": "key1"})
        endpoint2 = Endpoint(model="test", weight=1, params={"api_key": "key2"})
        lb.endpoint_configs["test-model"] = [endpoint1, endpoint2]

        with patch("random.uniform", return_value=1.5):  # Should select endpoint1
            result = await lb.select_endpoint("test-model")

        assert result == endpoint1
        mock_state_manager.get_availability_bulk.assert_called_once()
        mock_state_manager.refresh_ttl_bulk.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_endpoint_fallback_to_weight_zero(self):
        """Test selecting endpoint when primary endpoints are down, fallback to weight=0"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)

        # Primary endpoint unavailable, fallback available
        def mock_availability(endpoint_id):
            if "primary" in endpoint_id:
                return False
            return True

        mock_state_manager.get_availability_bulk.return_value = (
            {
                "primary-endpoint": False,
                "fallback-endpoint": True,
            },
            {
                "primary-endpoint": {},
                "fallback-endpoint": {},
            },
        )

        lb = LoadBalancer(state_manager=mock_state_manager)

        # Create test endpoints - primary with weight > 0, fallback with weight = 0
        primary_endpoint = Endpoint(
            model="test", weight=1, params={"api_key": "primary"}
        )
        primary_endpoint.id = "primary-endpoint"
        fallback_endpoint = Endpoint(
            model="test", weight=0, params={"api_key": "fallback"}
        )
        fallback_endpoint.id = "fallback-endpoint"

        lb.endpoint_configs["test-model"] = [primary_endpoint, fallback_endpoint]

        mock_state_manager.get_availability_bulk.return_value = (
            {
                primary_endpoint.id: False,
                fallback_endpoint.id: True,
            },
            {
                primary_endpoint.id: {},
                fallback_endpoint.id: {},
            },
        )

        result = await lb.select_endpoint("test-model")

        assert result == fallback_endpoint
        mock_state_manager.refresh_ttl_bulk.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_endpoint_no_available_endpoints(self):
        """Test selecting endpoint when no endpoints are available"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)
        mock_state_manager.get_availability_bulk.return_value = (
            {"test-endpoint": False},
            {"test-endpoint": {}},
        )

        lb = LoadBalancer(state_manager=mock_state_manager)

        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})
        lb.endpoint_configs["test-model"] = [endpoint]

        mock_state_manager.get_availability_bulk.return_value = (
            {endpoint.id: False},
            {endpoint.id: {}},
        )

        result = await lb.select_endpoint("test-model")

        assert result is None
        mock_state_manager.refresh_ttl_bulk.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_endpoint_without_state_manager(self):
        """Test selecting endpoint without state manager (all endpoints considered available)"""
        lb = LoadBalancer()

        endpoint1 = Endpoint(model="test", weight=2, params={"api_key": "key1"})
        endpoint2 = Endpoint(model="test", weight=1, params={"api_key": "key2"})
        lb.endpoint_configs["test-model"] = [endpoint1, endpoint2]

        with patch("random.uniform", return_value=1.5):  # Should select endpoint1
            result = await lb.select_endpoint("test-model")

        assert result == endpoint1

    @pytest.mark.asyncio
    async def test_select_endpoint_uniform_random_for_zero_weights(self):
        """Test uniform random selection when all weights are 0"""
        lb = LoadBalancer()

        endpoint1 = Endpoint(model="test", weight=0, params={"api_key": "key1"})
        endpoint2 = Endpoint(model="test", weight=0, params={"api_key": "key2"})
        lb.endpoint_configs["test-model"] = [endpoint1, endpoint2]

        with patch("random.choice", return_value=endpoint2):
            result = await lb.select_endpoint("test-model")

        assert result == endpoint2

    @pytest.mark.asyncio
    async def test_record_success(self):
        """Test recording successful request"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)
        lb = LoadBalancer(
            state_manager=mock_state_manager, allowed_fails=2, cooldown_time=120
        )

        # Set up endpoint configs for _get_model_group_for_endpoint
        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})
        lb.endpoint_configs["test-model"] = [endpoint]

        await lb.record_success(endpoint)

        mock_state_manager.ensure_endpoint_state.assert_called_once_with(
            endpoint, "test-model"
        )
        mock_state_manager.record_request_outcome.assert_called_once_with(
            endpoint.id, success=True, allowed_fails=2, cooldown_time=120
        )

    @pytest.mark.asyncio
    async def test_record_success_without_state_manager(self):
        """Test recording success without state manager"""
        lb = LoadBalancer()
        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})

        # Should not raise any exceptions
        await lb.record_success(endpoint)

    @pytest.mark.asyncio
    async def test_record_failure(self):
        """Test recording failed request"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)
        lb = LoadBalancer(
            state_manager=mock_state_manager, allowed_fails=2, cooldown_time=120
        )

        # Set up endpoint configs for _get_model_group_for_endpoint
        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})
        lb.endpoint_configs["test-model"] = [endpoint]

        await lb.record_failure(endpoint, "Connection timeout")

        mock_state_manager.ensure_endpoint_state.assert_called_once_with(
            endpoint, "test-model"
        )
        mock_state_manager.record_request_outcome.assert_called_once_with(
            endpoint.id,
            success=False,
            error="Connection timeout",
            allowed_fails=2,
            cooldown_time=120,
        )

    @pytest.mark.asyncio
    async def test_record_failure_without_state_manager(self):
        """Test recording failure without state manager"""
        lb = LoadBalancer()
        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})

        # Should not raise any exceptions
        await lb.record_failure(endpoint, "Error message")

    def test_get_model_group_for_endpoint(self):
        """Test getting model group for endpoint"""
        lb = LoadBalancer()

        endpoint1 = Endpoint(model="test", weight=1, params={"api_key": "key1"})
        endpoint2 = Endpoint(model="test", weight=1, params={"api_key": "key2"})

        # Ensure endpoints have different IDs
        endpoint1.id = "endpoint1"
        endpoint2.id = "endpoint2"

        lb.endpoint_configs["group1"] = [endpoint1]
        lb.endpoint_configs["group2"] = [endpoint2]

        result1 = lb._get_model_group_for_endpoint(endpoint1)
        result2 = lb._get_model_group_for_endpoint(endpoint2)

        assert result1 == "group1"
        assert result2 == "group2"

    def test_get_model_group_for_endpoint_not_found(self):
        """Test getting model group for endpoint that doesn't exist"""
        lb = LoadBalancer()
        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})

        result = lb._get_model_group_for_endpoint(endpoint)

        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_get_all_endpoints_stats(self):
        """Test getting all endpoints stats"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)

        lb = LoadBalancer(state_manager=mock_state_manager)

        endpoint1 = Endpoint(model="test1", weight=1, params={"api_key": "key1"})
        endpoint2 = Endpoint(model="test2", weight=2, params={"api_key": "key2"})

        lb.endpoint_configs["group1"] = [endpoint1]
        lb.endpoint_configs["group2"] = [endpoint2]

        mock_state_manager.get_endpoint_stats_bulk.return_value = {
            endpoint1.id: {
                "id": endpoint1.id,
                "model": endpoint1.model,
                "weight": endpoint1.weight,
                "status": "healthy",
            },
            endpoint2.id: {
                "id": endpoint2.id,
                "model": endpoint2.model,
                "weight": endpoint2.weight,
                "status": "cooling_down",
            },
        }

        result = await lb.get_all_endpoints_stats()

        assert "group1" in result
        assert "group2" in result
        assert len(result["group1"]) == 1
        assert len(result["group2"]) == 1

        stats1 = result["group1"][0]
        assert stats1["id"] == endpoint1.id
        assert stats1["config"]["model"] == endpoint1.model
        assert stats1["state"]["status"] == "healthy"
        mock_state_manager.get_endpoint_stats_bulk.assert_called()

    @pytest.mark.asyncio
    async def test_get_all_endpoints_stats_without_state_manager(self):
        """Test getting all endpoints stats without state manager"""
        lb = LoadBalancer()

        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})
        lb.endpoint_configs["group1"] = [endpoint]

        result = await lb.get_all_endpoints_stats()

        assert "group1" in result
        assert len(result["group1"]) == 1

        stats = result["group1"][0]
        # When no state manager, it returns the config dict from endpoint
        assert "model" in stats
        assert "weight" in stats

    def test_get_model_groups(self):
        """Test getting list of model groups"""
        lb = LoadBalancer()

        endpoint1 = Endpoint(model="test1", weight=1, params={"api_key": "key1"})
        endpoint2 = Endpoint(model="test2", weight=1, params={"api_key": "key2"})

        lb.endpoint_configs["group1"] = [endpoint1]
        lb.endpoint_configs["group2"] = [endpoint2]

        result = lb.get_model_groups()

        assert set(result) == {"group1", "group2"}

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutdown method"""
        lb = LoadBalancer()

        # Should not raise any exceptions
        await lb.shutdown()

    @pytest.mark.asyncio
    async def test_log_endpoint_states(self):
        """Test logging endpoint states"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)

        lb = LoadBalancer(state_manager=mock_state_manager)

        endpoint = Endpoint(model="test", weight=1, params={"api_key": "key"})

        availability = {endpoint.id: True}
        states = {endpoint.id: {"status": "healthy"}}

        await lb._log_endpoint_states([endpoint], "test-model", states, availability)

    @pytest.mark.asyncio
    async def test_get_available_endpoints(self):
        """Test getting available endpoints"""
        mock_state_manager = AsyncMock(spec=EndpointStateManager)

        # First endpoint available, second not available
        def mock_availability(endpoint_id):
            return endpoint_id == "endpoint1"

        mock_state_manager.get_availability_bulk.side_effect = lambda ids: (
            {endpoint_id: True for endpoint_id in ids},
            {endpoint_id: {"status": "healthy"} for endpoint_id in ids},
        )

        lb = LoadBalancer(state_manager=mock_state_manager)

        endpoint1 = Endpoint(model="test", weight=1, params={"api_key": "key1"})
        endpoint1.id = "endpoint1"
        endpoint2 = Endpoint(
            model="test", weight=0, params={"api_key": "key2"}
        )  # fallback
        endpoint2.id = "endpoint2"
        endpoint3 = Endpoint(model="test", weight=2, params={"api_key": "key3"})
        endpoint3.id = "endpoint3"

        configs = [endpoint1, endpoint2, endpoint3]

        availability = {
            endpoint1.id: True,
            endpoint2.id: False,
            endpoint3.id: True,
        }

        primary_available, fallback_available = lb._split_available_endpoints(
            configs, availability
        )

        # Only endpoint1 should be available and it has weight > 0
        assert len(primary_available) == 2
        assert primary_available[0] == endpoint1
        assert primary_available[1] == endpoint3

        # endpoint2 should be in fallback with weight 0 but only if available
        assert len(fallback_available) == 0
