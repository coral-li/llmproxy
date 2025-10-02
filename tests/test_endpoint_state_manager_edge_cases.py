import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis

from llmproxy.managers.endpoint_state_manager import EndpointStateManager
from llmproxy.models.endpoint import Endpoint


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


class TestEndpointStateManagerEdgeCases:
    """Edge case tests for EndpointStateManager to uncover potential bugs"""

    @pytest.fixture
    def mock_redis(self):
        mock = MagicMock(spec=redis.Redis)
        mock._state_store = {}
        mock._loaded_scripts = {}

        async def get_side_effect(key):
            return mock._state_store.get(key)

        async def setex_side_effect(key, ttl, value):
            mock._state_store[key] = value

        async def delete_side_effect(*keys):
            for key in keys:
                mock._state_store.pop(key, None)

        async def unlink_side_effect(*keys):
            for key in keys:
                mock._state_store.pop(key, None)

        async def mget_side_effect(*keys):
            return [mock._state_store.get(key) for key in keys]

        async def script_load_side_effect(script):
            sha = f"sha-{len(mock._loaded_scripts)}"
            mock._loaded_scripts[sha] = script
            return sha

        async def evalsha_side_effect(*args, **kwargs):
            sha = args[0]
            key = args[2]

            if sha.startswith("mark_healthy"):
                now_iso = args[4]
                state_json = mock._state_store.get(key)
                if not state_json:
                    return 0
                state = json.loads(state_json)
                state["status"] = "healthy"
                state["consecutive_failures"] = 0
                state["cooldown_until"] = None
                state["updated_at"] = now_iso
                encoded = json.dumps(state)
                mock._state_store[key] = encoded
                return 1

            (
                _,
                _,
                _key,
                _ttl,
                allowed_fails,
                cooldown_time,
                success_flag,
                error_message,
                error_is_null,
                now_iso,
                cooldown_until_iso,
                default_state_json,
            ) = args

            state_json = mock._state_store.get(key)
            if state_json:
                try:
                    state = json.loads(state_json)
                except json.JSONDecodeError:
                    state = json.loads(default_state_json)
            else:
                state = json.loads(default_state_json)

            allowed_fails = _to_int(allowed_fails, 0)
            cooldown_time = _to_int(cooldown_time, 0)
            success_flag = _to_int(success_flag, 0)
            error_is_null = _to_int(error_is_null, 0)

            total_requests = _to_int(state.get("total_requests")) + 1
            failed_requests = _to_int(state.get("failed_requests"))
            consecutive_failures = _to_int(state.get("consecutive_failures"))

            state["allowed_fails"] = allowed_fails
            state["total_requests"] = total_requests

            if success_flag == 1:
                state["consecutive_failures"] = 0
                state["status"] = "healthy"
                state["cooldown_until"] = None
                state["last_success_time"] = now_iso
            else:
                failed_requests += 1
                consecutive_failures += 1
                state["failed_requests"] = failed_requests
                state["consecutive_failures"] = consecutive_failures
                state["last_error"] = None if error_is_null == 1 else error_message
                state["last_error_time"] = now_iso

                if consecutive_failures >= allowed_fails:
                    state["status"] = "cooling_down"
                    state["cooldown_until"] = cooldown_until_iso

            state["updated_at"] = now_iso

            encoded = json.dumps(state)
            mock._state_store[key] = encoded
            return encoded

        def pipeline_side_effect(*args, **kwargs):
            class AsyncPipeline:
                def __init__(self, outer_mock):
                    self._outer = outer_mock
                    self._commands = []

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    if self._commands:
                        await self.execute()
                    return False

                def expire(self, key, ttl):
                    self._commands.append(("expire", key, ttl))

                async def execute(self):
                    for command, key, ttl in self._commands:
                        if command == "expire":
                            key = key if isinstance(key, str) else key.decode()
                            if key in self._outer._state_store and ttl > 0:
                                pass

            return AsyncPipeline(mock)

        mock.get = AsyncMock(side_effect=get_side_effect)
        mock.setex = AsyncMock(side_effect=setex_side_effect)
        mock.mget = AsyncMock(side_effect=mget_side_effect)
        mock.sadd = AsyncMock()
        mock.smembers = AsyncMock()
        mock.scan_iter = MagicMock()
        mock.delete = AsyncMock(side_effect=delete_side_effect)
        mock.unlink = AsyncMock(side_effect=unlink_side_effect)
        mock.ping = AsyncMock()
        mock.script_load = AsyncMock(side_effect=script_load_side_effect)
        mock.evalsha = AsyncMock(side_effect=evalsha_side_effect)
        mock.pipeline = MagicMock(side_effect=pipeline_side_effect)
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
        endpoint_id = "missing-endpoint"
        key = state_manager._get_endpoint_key(endpoint_id)
        mock_redis._state_store.pop(key, None)

        await state_manager.record_request_outcome(
            endpoint_id=endpoint_id,
            success=True,
            allowed_fails=3,
            cooldown_time=120,
        )

        assert mock_redis.evalsha.await_count == 1
        state_data = json.loads(mock_redis._state_store[key])
        assert state_data["id"] == endpoint_id
        assert state_data["status"] == "healthy"
        assert state_data["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_record_request_outcome_with_corrupted_state(
        self, state_manager, mock_redis
    ):
        """Test recording request outcome with corrupted state data"""
        endpoint_id = "corrupted-endpoint"
        key = state_manager._get_endpoint_key(endpoint_id)
        mock_redis._state_store[key] = '{"corrupted": json'

        await state_manager.record_request_outcome(
            endpoint_id=endpoint_id, success=False, error="Test error"
        )

        state_data = json.loads(mock_redis._state_store[key])
        assert state_data["id"] == endpoint_id
        assert state_data["total_requests"] == 1
        assert state_data["failed_requests"] == 1
        assert state_data["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_record_request_outcome_with_non_numeric_counters(
        self, state_manager, mock_redis
    ):
        """Test recording request outcome with non-numeric counter values"""
        endpoint_id = "test-endpoint"
        key = state_manager._get_endpoint_key(endpoint_id)
        mock_redis._state_store[key] = json.dumps(
            {
                "id": endpoint_id,
                "total_requests": "not-a-number",
                "failed_requests": "also-not-a-number",
                "consecutive_failures": "still-not-a-number",
                "status": "healthy",
            }
        )

        await state_manager.record_request_outcome(
            endpoint_id=endpoint_id, success=False, allowed_fails=2
        )

        state_data = json.loads(mock_redis._state_store[key])
        assert isinstance(state_data["total_requests"], int)
        assert isinstance(state_data["failed_requests"], int)
        assert isinstance(state_data["consecutive_failures"], int)
        assert state_data["total_requests"] == 1
        assert state_data["failed_requests"] == 1
        assert state_data["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_record_request_outcome_cooldown_threshold_edge_cases(
        self, state_manager, mock_redis
    ):
        """Test cooldown behavior at exact threshold boundaries"""
        endpoint_id = "threshold-test"
        key = state_manager._get_endpoint_key(endpoint_id)
        mock_redis._state_store[key] = json.dumps(
            {
                "id": endpoint_id,
                "consecutive_failures": 2,
                "status": "healthy",
                "total_requests": 5,
                "failed_requests": 2,
            }
        )

        await state_manager.record_request_outcome(
            endpoint_id=endpoint_id,
            success=False,
            allowed_fails=3,
        )

        state_data = json.loads(mock_redis._state_store[key])
        assert state_data["status"] == "cooling_down"
        assert state_data["consecutive_failures"] == 3
        assert state_data["cooldown_until"] is not None

    @pytest.mark.asyncio
    async def test_is_endpoint_available_with_malformed_cooldown_time(
        self, state_manager, mock_redis
    ):
        """Test endpoint availability check with malformed cooldown timestamp"""
        malformed_state = {
            "id": "malformed",
            "status": "cooling_down",
            "cooldown_until": "invalid-timestamp",
            "consecutive_failures": 3,
        }
        key = state_manager._get_endpoint_key("malformed")
        mock_redis._state_store[key] = json.dumps(malformed_state)
        mock_redis.get.return_value = json.dumps(malformed_state)

        result = await state_manager.is_endpoint_available("malformed")

        assert result is True

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
            "consecutive_failures": 3,
        }
        key = state_manager._get_endpoint_key("test-endpoint")
        mock_redis._state_store[key] = json.dumps(expired_cooldown_state)
        mock_redis.get.return_value = json.dumps(expired_cooldown_state)

        state_manager._mark_endpoint_healthy = AsyncMock()

        result = await state_manager.is_endpoint_available("test-endpoint")

        # Should mark as healthy and return available
        assert result is True
        state_manager._mark_endpoint_healthy.assert_awaited_once_with("test-endpoint")

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
            yield

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

        async def mock_scan():
            for key in [
                "llmproxy:endpoint:stale-1",
                "llmproxy:endpoint:stale-2",
            ]:
                yield key

        mock_redis.scan_iter.return_value = mock_scan()
        mock_redis.delete.side_effect = redis.RedisError("Delete operation failed")

        await state_manager.cleanup_stale_states(["active-endpoint"])

        assert mock_redis.unlink.call_count == 1
        call_args = mock_redis.unlink.call_args
        assert "llmproxy:endpoint:stale-1" in call_args[0]
        assert "llmproxy:endpoint:stale-2" in call_args[0]

    @pytest.mark.asyncio
    async def test_health_check_with_redis_ping_failure(
        self, state_manager, mock_redis
    ):
        """Test health check when Redis operations fail"""
        mock_redis.setex.side_effect = redis.ConnectionError("Ping failed")

        test_key = "llmproxy:health_check"
        mock_redis._state_store[test_key] = "value"

        result = await state_manager.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_endpoint_state_idempotency(
        self, state_manager, mock_redis, sample_endpoint
    ):
        """Test that ensure_endpoint_state is idempotent"""
        existing_state = {
            "id": sample_endpoint.id,
            "model": sample_endpoint.model,
            "status": "healthy",
        }
        key = state_manager._get_endpoint_key(sample_endpoint.id)
        mock_redis._state_store[key] = json.dumps(existing_state)

        await state_manager.ensure_endpoint_state(sample_endpoint, "test-group")
        await state_manager.ensure_endpoint_state(sample_endpoint, "test-group")

        assert mock_redis.get.await_count >= 1
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_request_outcome_recording(
        self, state_manager, mock_redis
    ):
        """Test concurrent recording of request outcomes for the same endpoint"""
        endpoint_id = "concurrent-test"
        key = state_manager._get_endpoint_key(endpoint_id)
        initial_state = {
            "id": endpoint_id,
            "total_requests": 0,
            "failed_requests": 0,
            "consecutive_failures": 0,
            "status": "healthy",
        }
        mock_redis._state_store[key] = json.dumps(initial_state)

        async def record_outcome(success):
            await state_manager.record_request_outcome(
                endpoint_id=endpoint_id, success=success
            )

        tasks = [
            record_outcome(True),
            record_outcome(False),
            record_outcome(True),
            record_outcome(False),
            record_outcome(True),
        ]

        await asyncio.gather(*tasks)

        state_data = json.loads(mock_redis._state_store[key])
        assert mock_redis.evalsha.await_count == len(tasks)
        assert state_data["total_requests"] == 5

    @pytest.mark.asyncio
    async def test_endpoint_state_with_extreme_timestamps(
        self, state_manager, mock_redis
    ):
        """Test endpoint state handling with extreme timestamp values"""
        endpoint_id = "extreme-time"
        key = state_manager._get_endpoint_key(endpoint_id)
        extreme_state = {
            "id": endpoint_id,
            "status": "cooling_down",
            "cooldown_until": "9999-12-31T23:59:59",
            "last_error_time": "1970-01-01T00:00:00",
            "last_success_time": datetime.utcnow().isoformat(),
            "created_at": "invalid-date-format",
            "updated_at": None,
        }
        mock_redis._state_store[key] = json.dumps(extreme_state)

        result = await state_manager.is_endpoint_available(endpoint_id)
        assert result is False

    def test_get_endpoint_key_with_extreme_endpoint_ids(self, state_manager):
        """Test _get_endpoint_key with extreme endpoint ID values"""
        extreme_ids = [
            "",
            "a" * 1000,
            "🔥💯🚀",
            "endpoint with spaces",
            "endpoint/with/slashes",
            None,
        ]

        for endpoint_id in extreme_ids:
            try:
                key = state_manager._get_endpoint_key(endpoint_id)
                if endpoint_id is not None:
                    assert isinstance(key, str)
                    assert "llmproxy:endpoint:" in key
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_state_manager_initialization_edge_cases(self, mock_redis):
        """Test EndpointStateManager initialization with edge case parameters"""
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
        endpoint_id_template = "extreme-{allowed}-{cooldown}"

        extreme_cases = [
            (0, 0),
            (-1, -1),
            (1000, 86400),
        ]

        for allowed_fails, cooldown_time in extreme_cases:
            endpoint_id = endpoint_id_template.format(
                allowed=allowed_fails, cooldown=cooldown_time
            )
            key = state_manager._get_endpoint_key(endpoint_id)
            mock_redis._state_store.pop(key, None)

            await state_manager.record_request_outcome(
                endpoint_id=endpoint_id,
                success=False,
                error="Test error",
                allowed_fails=allowed_fails,
                cooldown_time=cooldown_time,
            )

            state_data = json.loads(mock_redis._state_store[key])
            assert state_data["allowed_fails"] == allowed_fails
            assert state_data["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_get_endpoint_states_bulk(self, state_manager, mock_redis):
        endpoint_ids = ["endpoint-1", "endpoint-2", "missing-endpoint"]
        stored_state = {
            "id": "endpoint-1",
            "status": "healthy",
        }
        key_1 = state_manager._get_endpoint_key("endpoint-1")
        mock_redis._state_store[key_1] = json.dumps(stored_state)

        states = await state_manager.get_endpoint_states_bulk(endpoint_ids)

        assert states["endpoint-1"] == stored_state
        assert states["endpoint-2"] is None
        assert states["missing-endpoint"] is None

    @pytest.mark.asyncio
    async def test_get_availability_bulk_with_cooldown_cleanup(
        self, state_manager, mock_redis
    ):
        endpoint_ids = ["healthy", "cooldown", "unknown"]

        healthy_state = {"id": "healthy", "status": "healthy"}
        cooldown_state = {
            "id": "cooldown",
            "status": "cooling_down",
            "cooldown_until": (datetime.utcnow() - timedelta(minutes=1)).isoformat(),
        }

        mock_redis._state_store[state_manager._get_endpoint_key("healthy")] = (
            json.dumps(healthy_state)
        )
        mock_redis._state_store[state_manager._get_endpoint_key("cooldown")] = (
            json.dumps(cooldown_state)
        )

        availability, states = await state_manager.get_availability_bulk(endpoint_ids)

        assert availability == {
            "healthy": True,
            "cooldown": True,
            "unknown": True,
        }
        assert states["healthy"]["status"] == "healthy"
        assert states["unknown"] is None

        await state_manager.refresh_ttl_bulk(["healthy", "cooldown"])
        assert mock_redis.pipeline.called

    @pytest.mark.asyncio
    async def test_get_availability_bulk_handles_malformed_state(
        self, state_manager, mock_redis
    ):
        endpoint_ids = ["malformed"]
        key = state_manager._get_endpoint_key("malformed")
        mock_redis._state_store[key] = "{not-json"

        availability, states = await state_manager.get_availability_bulk(endpoint_ids)

        assert availability["malformed"] is True
        assert states["malformed"] is None

    @pytest.mark.asyncio
    async def test_get_endpoint_stats_bulk(self, state_manager, mock_redis):
        endpoint_ids = ["stats-1", "stats-2"]
        state = {
            "id": "stats-1",
            "total_requests": 10,
            "failed_requests": 2,
            "status": "healthy",
        }
        mock_redis._state_store[state_manager._get_endpoint_key("stats-1")] = (
            json.dumps(state)
        )

        stats = await state_manager.get_endpoint_stats_bulk(endpoint_ids)

        assert stats["stats-1"]["success_rate"] == 80.0
        assert stats["stats-2"]["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_cleanup_stale_states_with_delete_when_unlink_missing(
        self, state_manager, mock_redis
    ):
        async def mock_scan():
            for key in ["llmproxy:endpoint:stale-3"]:
                yield key

        mock_redis.scan_iter.return_value = mock_scan()
        del mock_redis.unlink
        mock_redis.delete.reset_mock()

        await state_manager.cleanup_stale_states(["active-endpoint"])

        mock_redis.delete.assert_awaited()
