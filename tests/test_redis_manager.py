"""Tests for the Redis manager module"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import redis.asyncio as redis

from llmproxy.core.redis_manager import RedisManager


class TestRedisManager:
    """Test cases for RedisManager class"""

    def test_redis_manager_initialization(self):
        """Test RedisManager initialization"""
        manager = RedisManager(host="localhost", port=6379, password="secret")

        assert manager.host == "localhost"
        assert manager.port == 6379
        assert manager.password == "secret"
        assert manager.client is None
        assert manager._pool is None

    def test_redis_manager_initialization_with_string_port(self):
        """Test RedisManager initialization with string port"""
        manager = RedisManager(host="redis.example.com", port="6380", password=None)

        assert manager.host == "redis.example.com"
        assert manager.port == 6380
        assert manager.password is None

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful Redis connection"""
        manager = RedisManager(host="localhost", port=6379)

        with patch("redis.asyncio.ConnectionPool") as mock_pool_class, patch(
            "redis.asyncio.Redis"
        ) as mock_redis_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool

            mock_client = AsyncMock()
            mock_redis_class.return_value = mock_client

            await manager.connect()

            # Verify connection pool creation
            mock_pool_class.assert_called_once_with(
                host="localhost",
                port=6379,
                password=None,
                decode_responses=True,
                max_connections=50,
            )

            # Verify Redis client creation
            mock_redis_class.assert_called_once_with(connection_pool=mock_pool)

            # Verify ping was called
            mock_client.ping.assert_called_once()

            assert manager.client == mock_client
            assert manager._pool == mock_pool

    @pytest.mark.asyncio
    async def test_connect_with_password(self):
        """Test Redis connection with password"""
        manager = RedisManager(host="localhost", port=6379, password="secret123")

        with patch("redis.asyncio.ConnectionPool") as mock_pool_class, patch(
            "redis.asyncio.Redis"
        ) as mock_redis_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool

            mock_client = AsyncMock()
            mock_redis_class.return_value = mock_client

            await manager.connect()

            # Verify connection pool creation with password
            mock_pool_class.assert_called_once_with(
                host="localhost",
                port=6379,
                password="secret123",
                decode_responses=True,
                max_connections=50,
            )

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test Redis connection failure"""
        manager = RedisManager(host="localhost", port=6379)

        with patch("redis.asyncio.ConnectionPool"), patch(
            "redis.asyncio.Redis"
        ) as mock_redis_class:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = redis.ConnectionError("Connection failed")
            mock_redis_class.return_value = mock_client

            with pytest.raises(redis.ConnectionError):
                await manager.connect()

    @pytest.mark.asyncio
    async def test_disconnect_with_client_and_pool(self):
        """Test Redis disconnection when both client and pool exist"""
        manager = RedisManager(host="localhost", port=6379)

        mock_client = AsyncMock()
        mock_pool = AsyncMock()

        manager.client = mock_client
        manager._pool = mock_pool

        await manager.disconnect()

        mock_client.aclose.assert_called_once()
        mock_pool.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_with_only_client(self):
        """Test Redis disconnection when only client exists"""
        manager = RedisManager(host="localhost", port=6379)

        mock_client = AsyncMock()
        manager.client = mock_client
        manager._pool = None

        await manager.disconnect()

        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_with_no_client(self):
        """Test Redis disconnection when no client exists"""
        manager = RedisManager(host="localhost", port=6379)

        # Should not raise any exceptions
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check"""
        manager = RedisManager(host="localhost", port=6379)

        mock_client = AsyncMock()
        manager.client = mock_client

        result = await manager.health_check()

        assert result is True
        mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_no_client(self):
        """Test health check when no client exists"""
        manager = RedisManager(host="localhost", port=6379)

        result = await manager.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_ping_failure(self):
        """Test health check when ping fails"""
        manager = RedisManager(host="localhost", port=6379)

        mock_client = AsyncMock()
        mock_client.ping.side_effect = redis.ConnectionError("Connection lost")
        manager.client = mock_client

        result = await manager.health_check()

        assert result is False
        mock_client.ping.assert_called_once()

    def test_get_client_success(self):
        """Test getting client when it exists"""
        manager = RedisManager(host="localhost", port=6379)

        mock_client = Mock()
        manager.client = mock_client

        result = manager.get_client()

        assert result == mock_client

    def test_get_client_not_initialized(self):
        """Test getting client when not initialized"""
        manager = RedisManager(host="localhost", port=6379)

        with pytest.raises(RuntimeError, match="Redis client not initialized"):
            manager.get_client()
