import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.models.endpoint import Endpoint, EndpointStatus
from config_model import LLMProxyConfig, ModelGroup, ModelConfig, GeneralSettings


class TestLoadBalancer:
    """Tests for the load balancer"""

    @pytest.fixture
    def load_balancer(self):
        """Create a load balancer instance"""
        return LoadBalancer(cooldown_time=60, allowed_fails=2)

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        return LLMProxyConfig(
            model_groups=[
                ModelGroup(
                    model_group="test-model",
                    models=[
                        ModelConfig(
                            model="test-model",
                            weight=2,
                            params={"api_key": "key1", "base_url": "http://endpoint1"},
                        ),
                        ModelConfig(
                            model="test-model",
                            weight=1,
                            params={"api_key": "key2", "base_url": "http://endpoint2"},
                        ),
                        ModelConfig(
                            model="test-model",
                            weight=0,  # Fallback
                            params={"api_key": "key3", "base_url": "http://endpoint3"},
                        ),
                    ],
                )
            ],
            general_settings=GeneralSettings(
                redis_host="localhost",
                redis_port=6379,
                redis_password="",
                allowed_fails=2,
                cooldown_time=60,
            ),
        )

    @pytest.mark.asyncio
    async def test_initialize_from_config(self, load_balancer, mock_config):
        """Test initializing load balancer from config"""
        await load_balancer.initialize_from_config(mock_config)

        assert "test-model" in load_balancer.endpoint_pools
        pool = load_balancer.endpoint_pools["test-model"]
        assert len(pool) == 3
        assert pool[0].weight == 2
        assert pool[1].weight == 1
        assert pool[2].weight == 0

    @pytest.mark.asyncio
    async def test_weighted_selection(self, load_balancer, mock_config):
        """Test weighted endpoint selection"""
        await load_balancer.initialize_from_config(mock_config)

        # Run many selections to verify weight distribution
        selections = {
            "http://endpoint1": 0,
            "http://endpoint2": 0,
            "http://endpoint3": 0,
        }

        for _ in range(1000):
            endpoint = await load_balancer.select_endpoint("test-model")
            assert endpoint is not None
            selections[endpoint.base_url] += 1

        # With weights 2:1:0, endpoint1 should get ~66%, endpoint2 ~33%, endpoint3 0%
        assert selections["http://endpoint1"] > 600
        assert selections["http://endpoint2"] > 300
        assert (
            selections["http://endpoint3"] == 0
        )  # Weight 0 should not be selected normally

    @pytest.mark.asyncio
    async def test_fallback_selection(self, load_balancer, mock_config):
        """Test fallback to weight=0 endpoints when all others fail"""
        await load_balancer.initialize_from_config(mock_config)

        # Mark all non-zero weight endpoints as failed
        pool = load_balancer.endpoint_pools["test-model"]
        for endpoint in pool:
            if endpoint.weight > 0:
                endpoint.status = EndpointStatus.COOLING_DOWN
                endpoint.cooldown_until = datetime.utcnow() + timedelta(minutes=5)

        # Should now select the weight=0 fallback
        endpoint = await load_balancer.select_endpoint("test-model")
        assert endpoint is not None
        assert endpoint.weight == 0
        assert endpoint.base_url == "http://endpoint3"

    @pytest.mark.asyncio
    async def test_no_available_endpoints(self, load_balancer, mock_config):
        """Test behavior when no endpoints are available"""
        await load_balancer.initialize_from_config(mock_config)

        # Mark all endpoints as failed
        pool = load_balancer.endpoint_pools["test-model"]
        for endpoint in pool:
            endpoint.status = EndpointStatus.COOLING_DOWN
            endpoint.cooldown_until = datetime.utcnow() + timedelta(minutes=5)

        # Should return None
        endpoint = await load_balancer.select_endpoint("test-model")
        assert endpoint is None

    def test_record_success(self, load_balancer):
        """Test recording successful requests"""
        endpoint = Endpoint(model="test", weight=1, params={}, allowed_fails=2)
        endpoint.consecutive_failures = 1

        load_balancer.record_success(endpoint)

        assert endpoint.consecutive_failures == 0
        assert endpoint.status == EndpointStatus.HEALTHY
        assert endpoint.total_requests == 1
        assert endpoint.failed_requests == 0

    def test_record_failure_with_cooldown(self, load_balancer):
        """Test recording failures and entering cooldown"""
        endpoint = Endpoint(model="test", weight=1, params={}, allowed_fails=2)

        # First failure
        load_balancer.record_failure(endpoint, "Error 1")
        assert endpoint.consecutive_failures == 1
        assert endpoint.status == EndpointStatus.HEALTHY

        # Second failure - should trigger cooldown
        load_balancer.record_failure(endpoint, "Error 2")
        assert endpoint.consecutive_failures == 2
        assert endpoint.status == EndpointStatus.COOLING_DOWN
        assert endpoint.cooldown_until > datetime.utcnow()

    def test_cooldown_recovery(self):
        """Test endpoint recovery from cooldown"""
        endpoint = Endpoint(model="test", weight=1, params={}, allowed_fails=1)

        # Put in cooldown
        endpoint.status = EndpointStatus.COOLING_DOWN
        endpoint.cooldown_until = datetime.utcnow() - timedelta(
            seconds=1
        )  # Already expired
        endpoint.consecutive_failures = 1

        # Check availability
        assert endpoint.is_available() is True
        assert endpoint.status == EndpointStatus.HEALTHY
        assert endpoint.consecutive_failures == 0
