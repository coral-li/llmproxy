import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis

from llmproxy.managers.endpoint_state_manager import EndpointStateManager
from llmproxy.models.endpoint import Endpoint


class TestEndpointStateManagerEdgeCases:
    """Edge case tests for EndpointStateManager to uncover potential bugs"""

    @pytest.fixture
    def mock_redis(self):
        mock = MagicMock(spec=redis.Redis)
        # Make Redis methods async
        mock.get = AsyncMock()
        mock.setex = AsyncMock()
        mock.sadd = AsyncMock()
        mock.smembers = AsyncMock()
        # scan_iter needs special handling to return an async iterator
        mock.scan_iter = MagicMock()
        mock.delete = AsyncMock()
        mock.ping = AsyncMock()
        return mock

    @pytest.fixture
    def state_manager(self, mock_redis):
        return EndpointStateManager(mock_redis, state_ttl=3600)

    @pytest.fixture
    def sample_endpoint(self):
        return Endpoint(
            model="gpt-3.5-turbo",
            weight=1,
            params={"base_url": "https://api.openai.com", "api_key": "test-key"},
            allowed_fails=1,
        )

    @pytest.mark.asyncio
    async def test_initialize_endpoint_with_none_values(
        self, state_manager, mock_redis
    ):
        """Test endpoint initialization with None/missing values"""
        # Mock Redis to return None (no existing state)
        mock_redis.get.return_value = None

        # Create endpoint with minimal data
        endpoint = Endpoint(
            model=None, weight=1, params={"base_url": None, "api_key": None}
        )

        await state_manager.initialize_endpoint(endpoint, "test-group")

        # Should handle None values gracefully
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        state_data = json.loads(call_args[0][2])
        assert state_data["model"] is None
        assert state_data["base_url"] is None

    @pytest.mark.asyncio
    async def test_initialize_endpoint_with_redis_failure(
        self, state_manager, mock_redis, sample_endpoint
    ):
        """Test endpoint initialization when Redis fails"""
        mock_redis.setex.side_effect = redis.RedisError("Redis connection failed")

        # Should not raise exception
        await state_manager.initialize_endpoint(sample_endpoint, "test-group")
        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_record_request_outcome_with_missing_state(
        self, state_manager, mock_redis
    ):
        """Test recording request outcome when endpoint state is missing"""
        mock_redis.get.return_value = None  # No existing state

        await state_manager.record_request_outcome(
            endpoint_id="missing-endpoint",
            success=True,
            allowed_fails=3,
            cooldown_time=120,
        )

        # Should create default state
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        state_data = json.loads(call_args[0][2])
        assert state_data["id"] == "missing-endpoint"
        assert state_data["status"] == "healthy"
        assert state_data["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_record_request_outcome_with_corrupted_state(
        self, state_manager, mock_redis
    ):
        """Test recording request outcome with corrupted state data"""
        mock_redis.get.return_value = '{"corrupted": json'  # Invalid JSON

        await state_manager.record_request_outcome(
            endpoint_id="corrupted-endpoint", success=False, error="Test error"
        )

        # Should create new state when unable to parse existing
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        state_data = json.loads(call_args[0][2])
        # Should have created default state with updated counters
        assert state_data["id"] == "corrupted-endpoint"
        assert state_data["total_requests"] == 1
        assert state_data["failed_requests"] == 1
        assert state_data["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_record_request_outcome_with_non_numeric_counters(
        self, state_manager, mock_redis
    ):
        """Test recording request outcome with non-numeric counter values"""
        corrupted_state = {
            "id": "test-endpoint",
            "total_requests": "not-a-number",
            "failed_requests": "also-not-a-number",
            "consecutive_failures": "still-not-a-number",
            "status": "healthy",
        }
        mock_redis.get.return_value = json.dumps(corrupted_state)

        await state_manager.record_request_outcome(
            endpoint_id="test-endpoint", success=False, allowed_fails=2
        )

        # Should handle non-numeric values by converting to 0
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        state_data = json.loads(call_args[0][2])
        assert isinstance(state_data["total_requests"], int)
        assert isinstance(state_data["failed_requests"], int)
        assert isinstance(state_data["consecutive_failures"], int)
        # Should treat non-numeric as 0 and then increment
        assert state_data["total_requests"] == 1
        assert state_data["failed_requests"] == 1
        assert state_data["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_record_request_outcome_cooldown_threshold_edge_cases(
        self, state_manager, mock_redis
    ):
        """Test cooldown behavior at exact threshold boundaries"""
        initial_state = {
            "id": "threshold-test",
            "consecutive_failures": 2,  # One less than threshold
            "status": "healthy",
            "total_requests": 5,
            "failed_requests": 2,
        }
        mock_redis.get.return_value = json.dumps(initial_state)

        # Record failure that should trigger cooldown (3rd consecutive failure)
        await state_manager.record_request_outcome(
            endpoint_id="threshold-test",
            success=False,
            allowed_fails=3,  # Exactly at threshold
        )

        call_args = mock_redis.setex.call_args
        state_data = json.loads(call_args[0][2])
        assert state_data["status"] == "cooling_down"
        assert state_data["consecutive_failures"] == 3
        assert "cooldown_until" in state_data

    @pytest.mark.asyncio
    async def test_is_endpoint_available_with_malformed_cooldown_time(
        self, state_manager, mock_redis
    ):
        """Test endpoint availability check with malformed cooldown timestamp"""
        malformed_state = {
            "id": "test-endpoint",
            "status": "cooling_down",
            "cooldown_until": "not-a-valid-timestamp",
        }
        mock_redis.get.return_value = json.dumps(malformed_state)

        # Mock the _mark_endpoint_healthy method
        state_manager._mark_endpoint_healthy = AsyncMock()

        # Should handle malformed timestamp gracefully
        result = await state_manager.is_endpoint_available("test-endpoint")
        # Should treat malformed timestamp as expired cooldown and mark as healthy
        assert result is True
        state_manager._mark_endpoint_healthy.assert_called_once_with("test-endpoint")

    @pytest.mark.asyncio
    async def test_is_endpoint_available_with_expired_cooldown(
        self, state_manager, mock_redis
    ):
        """Test endpoint availability when cooldown has expired"""
        past_time = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
        expired_cooldown_state = {
            "id": "test-endpoint",
            "status": "cooling_down",
            "cooldown_until": past_time,
        }
        mock_redis.get.return_value = json.dumps(expired_cooldown_state)

        # Mock the _mark_endpoint_healthy method
        state_manager._mark_endpoint_healthy = AsyncMock()

        result = await state_manager.is_endpoint_available("test-endpoint")

        # Should mark as healthy and return available
        assert result is True
        state_manager._mark_endpoint_healthy.assert_called_once_with("test-endpoint")

    @pytest.mark.asyncio
    async def test_is_endpoint_available_with_redis_error(
        self, state_manager, mock_redis
    ):
        """Test endpoint availability check when Redis fails"""
        mock_redis.get.side_effect = redis.ConnectionError("Redis connection failed")

        result = await state_manager.is_endpoint_available("test-endpoint")

        # Should default to available on error
        assert result is True

    @pytest.mark.asyncio
    async def test_mark_endpoint_healthy_with_missing_state(
        self, state_manager, mock_redis
    ):
        """Test marking endpoint healthy when state doesn't exist"""
        mock_redis.get.return_value = None  # No existing state

        await state_manager._mark_endpoint_healthy("missing-endpoint")

        # Should not crash, just handle gracefully
        # No setex should be called since there's no state to update
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_mark_endpoint_healthy_with_redis_error(
        self, state_manager, mock_redis
    ):
        """Test marking endpoint healthy when Redis operations fail"""
        mock_redis.get.return_value = json.dumps(
            {"id": "test", "status": "cooling_down"}
        )
        mock_redis.setex.side_effect = redis.RedisError("Redis write failed")

        # Should handle Redis error gracefully
        await state_manager._mark_endpoint_healthy("test-endpoint")
        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_get_endpoint_state_with_non_dict_data(
        self, state_manager, mock_redis
    ):
        """Test getting endpoint state when Redis returns non-dict data"""
        mock_redis.get.return_value = '"string instead of dict"'

        result = await state_manager.get_endpoint_state("test-endpoint")

        # Should return None for non-dict data
        assert result is None

    @pytest.mark.asyncio
    async def test_get_endpoint_stats_with_missing_state(self, state_manager):
        """Test getting endpoint stats when state doesn't exist"""
        # Mock get_endpoint_state to return None
        state_manager.get_endpoint_state = AsyncMock(return_value=None)

        stats = await state_manager.get_endpoint_stats("missing-endpoint")

        # Should return default stats
        assert stats["id"] == "missing-endpoint"
        assert stats["status"] == "unknown"
        assert stats["total_requests"] == 0
        assert stats["success_rate"] == 0

    @pytest.mark.asyncio
    async def test_get_endpoint_stats_with_zero_requests(self, state_manager):
        """Test endpoint stats calculation with zero total requests"""
        zero_requests_state = {
            "id": "test-endpoint",
            "total_requests": 0,
            "failed_requests": 0,
            "status": "healthy",
        }
        state_manager.get_endpoint_state = AsyncMock(return_value=zero_requests_state)

        stats = await state_manager.get_endpoint_stats("test-endpoint")

        # Should handle division by zero gracefully
        assert stats["success_rate"] == 0

    @pytest.mark.asyncio
    async def test_register_endpoint_in_group_concurrent_access(
        self, state_manager, mock_redis
    ):
        """Test concurrent registration of endpoints in the same group"""
        mock_redis.sadd.return_value = 1  # Simulate successful add

        async def register_endpoint(endpoint_id):
            await state_manager.register_endpoint_in_group("test-group", endpoint_id)

        # Register multiple endpoints concurrently
        tasks = [register_endpoint(f"endpoint-{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        # All registrations should complete without error
        assert mock_redis.sadd.call_count == 10

    @pytest.mark.asyncio
    async def test_get_group_endpoints_with_redis_error(
        self, state_manager, mock_redis
    ):
        """Test getting group endpoints when Redis fails"""
        mock_redis.smembers.side_effect = redis.ConnectionError(
            "Redis connection failed"
        )

        result = await state_manager.get_group_endpoints("test-group")

        # Should return empty list on error
        assert result == []

    @pytest.mark.asyncio
    async def test_cleanup_stale_states_with_scan_failure(
        self, state_manager, mock_redis
    ):
        """Test cleanup of stale states when Redis scan fails"""

        # Make scan_iter raise an error when called
        async def mock_scan_error(*args, **kwargs):
            raise redis.RedisError("Scan operation failed")
            # This is needed to make it an async generator
            yield  # pragma: no cover

        mock_redis.scan_iter.return_value = mock_scan_error()

        # Should handle scan failure gracefully
        await state_manager.cleanup_stale_states(
            ["active-endpoint-1", "active-endpoint-2"]
        )
        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_cleanup_stale_states_with_delete_failure(
        self, state_manager, mock_redis
    ):
        """Test cleanup when batch key deletion fails"""

        # Mock scan to return some stale keys (async iterator)
        async def mock_scan():
            for key in ["llmproxy:endpoint:stale-1", "llmproxy:endpoint:stale-2"]:
                yield key

        mock_redis.scan_iter.return_value = mock_scan()

        # Mock delete to fail
        mock_redis.delete.side_effect = redis.RedisError("Delete operation failed")

        # Should handle delete failure gracefully
        await state_manager.cleanup_stale_states(["active-endpoint"])

        # Should have attempted to delete both keys in one call
        assert mock_redis.delete.call_count == 1
        # Check that it tried to delete both stale keys
        call_args = mock_redis.delete.call_args
        assert "llmproxy:endpoint:stale-1" in call_args[0]
        assert "llmproxy:endpoint:stale-2" in call_args[0]

    @pytest.mark.asyncio
    async def test_health_check_with_redis_ping_failure(
        self, state_manager, mock_redis
    ):
        """Test health check when Redis ping fails"""
        mock_redis.ping.side_effect = redis.ConnectionError("Ping failed")

        result = await state_manager.health_check()

        # Should return False when ping fails
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_endpoint_state_idempotency(
        self, state_manager, mock_redis, sample_endpoint
    ):
        """Test that ensure_endpoint_state is idempotent"""
        # Mock that endpoint already exists
        existing_state = {
            "id": sample_endpoint.id,
            "model": sample_endpoint.model,
            "status": "healthy",
        }
        mock_redis.get.return_value = json.dumps(existing_state)

        # Call ensure multiple times
        await state_manager.ensure_endpoint_state(sample_endpoint, "test-group")
        await state_manager.ensure_endpoint_state(sample_endpoint, "test-group")

        # Should not reinitialize existing endpoint
        # Only initial check, no setex calls for existing state
        assert mock_redis.get.call_count >= 1
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_request_outcome_recording(
        self, state_manager, mock_redis
    ):
        """Test concurrent recording of request outcomes for the same endpoint"""
        initial_state = {
            "id": "concurrent-test",
            "total_requests": 0,
            "failed_requests": 0,
            "consecutive_failures": 0,
            "status": "healthy",
        }
        mock_redis.get.return_value = json.dumps(initial_state)

        async def record_outcome(success):
            await state_manager.record_request_outcome(
                endpoint_id="concurrent-test", success=success
            )

        # Record multiple outcomes concurrently
        tasks = [
            record_outcome(True),
            record_outcome(False),
            record_outcome(True),
            record_outcome(False),
            record_outcome(True),
        ]

        await asyncio.gather(*tasks)

        # All recordings should complete without error
        assert mock_redis.setex.call_count == 5

    @pytest.mark.asyncio
    async def test_endpoint_state_with_extreme_timestamps(
        self, state_manager, mock_redis
    ):
        """Test endpoint state handling with extreme timestamp values"""
        extreme_state = {
            "id": "extreme-time",
            "status": "cooling_down",
            "cooldown_until": "9999-12-31T23:59:59",  # Far future
            "last_error_time": "1970-01-01T00:00:00",  # Unix epoch
            "last_success_time": datetime.utcnow().isoformat(),
            "created_at": "invalid-date-format",
            "updated_at": None,
        }
        mock_redis.get.return_value = json.dumps(extreme_state)

        # Should handle extreme timestamps gracefully
        result = await state_manager.is_endpoint_available("extreme-time")
        assert result is False  # Should be unavailable due to far future cooldown

    def test_get_endpoint_key_with_extreme_endpoint_ids(self, state_manager):
        """Test _get_endpoint_key with extreme endpoint ID values"""
        extreme_ids = [
            "",  # Empty string
            "a" * 1000,  # Very long ID
            "🔥💯🚀",  # Unicode emojis
            "endpoint with spaces",  # Spaces
            "endpoint/with/slashes",  # Special characters
            None,  # None value (should handle gracefully)
        ]

        for endpoint_id in extreme_ids:
            try:
                key = state_manager._get_endpoint_key(endpoint_id)
                if endpoint_id is not None:
                    assert isinstance(key, str)
                    assert "llmproxy:endpoint:" in key
            except Exception:
                # Some extreme cases might raise exceptions, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_state_manager_initialization_edge_cases(self, mock_redis):
        """Test EndpointStateManager initialization with edge case parameters"""
        # Test with extreme TTL values
        manager1 = EndpointStateManager(mock_redis, state_ttl=0)
        assert manager1.state_ttl == 0

        manager2 = EndpointStateManager(mock_redis, state_ttl=-1)
        assert manager2.state_ttl == -1

        manager3 = EndpointStateManager(mock_redis, state_ttl=2**31)
        assert manager3.state_ttl == 2**31

    @pytest.mark.asyncio
    async def test_record_request_outcome_with_extreme_parameters(
        self, state_manager, mock_redis
    ):
        """Test recording request outcomes with extreme parameter values"""
        mock_redis.get.return_value = None  # No existing state

        # Test with extreme allowed_fails and cooldown_time values
        extreme_cases = [
            (0, 0),  # Zero values
            (-1, -1),  # Negative values
            (1000, 86400),  # Very large values
        ]

        for allowed_fails, cooldown_time in extreme_cases:
            await state_manager.record_request_outcome(
                endpoint_id=f"extreme-{allowed_fails}-{cooldown_time}",
                success=False,
                error="Test error",
                allowed_fails=allowed_fails,
                cooldown_time=cooldown_time,
            )

            # Should handle extreme values without crashing
            assert mock_redis.setex.called
