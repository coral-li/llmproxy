import pytest
import json
import hashlib
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llmproxy.core.cache_manager import CacheManager


class TestCacheManager:
    """Tests for the cache manager"""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client"""
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.setex = AsyncMock()
        return mock

    @pytest.fixture
    def cache_manager(self, mock_redis):
        """Create a cache manager with mock Redis"""
        return CacheManager(redis_client=mock_redis, ttl=300, namespace="test")

    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation is deterministic"""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 100,
        }

        key1 = cache_manager._generate_cache_key(request_data)
        key2 = cache_manager._generate_cache_key(request_data)

        # Same input should produce same key
        assert key1 == key2
        assert key1.startswith("test:")

        # Different input should produce different key
        request_data["temperature"] = 0.5
        key3 = cache_manager._generate_cache_key(request_data)
        assert key3 != key1

    def test_should_cache_logic(self, cache_manager):
        """Test cache decision logic"""
        # Should cache: deterministic request
        assert (
            cache_manager._should_cache({"temperature": 0, "stream": False, "n": 1})
            is True
        )

        # Should not cache: streaming
        assert cache_manager._should_cache({"temperature": 0, "stream": True}) is False

        # Should not cache: temperature > 0
        assert (
            cache_manager._should_cache({"temperature": 0.7, "stream": False}) is False
        )

        # Should not cache: multiple completions
        assert cache_manager._should_cache({"temperature": 0, "n": 3}) is False

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, cache_manager, mock_redis):
        """Test cache miss"""
        mock_redis.get.return_value = None

        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }

        result = await cache_manager.get(request_data)

        assert result is None
        assert cache_manager._misses == 1
        assert cache_manager._hits == 0
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get_hit(self, cache_manager, mock_redis):
        """Test cache hit"""
        cached_data = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_redis.get.return_value = json.dumps(cached_data)

        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }

        result = await cache_manager.get(request_data)

        assert result == cached_data
        assert cache_manager._hits == 1
        assert cache_manager._misses == 0

    @pytest.mark.asyncio
    async def test_cache_set(self, cache_manager, mock_redis):
        """Test setting cache"""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }

        response_data = {"choices": [{"message": {"content": "Hello!"}}]}

        await cache_manager.set(request_data, response_data)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # TTL
        assert json.loads(call_args[0][2]) == response_data

    @pytest.mark.asyncio
    async def test_cache_set_skip_streaming(self, cache_manager, mock_redis):
        """Test that streaming responses are not cached"""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        response_data = {"choices": [{"message": {"content": "Hello!"}}]}

        await cache_manager.set(request_data, response_data)

        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_set_skip_errors(self, cache_manager, mock_redis):
        """Test that error responses are not cached"""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }

        response_data = {"error": {"message": "Something went wrong"}}

        await cache_manager.set(request_data, response_data)

        mock_redis.setex.assert_not_called()

    def test_cache_stats(self, cache_manager):
        """Test cache statistics"""
        cache_manager._hits = 75
        cache_manager._misses = 25

        stats = cache_manager.get_stats()

        assert stats["hits"] == 75
        assert stats["misses"] == 25
        assert stats["total"] == 100
        assert stats["hit_rate"] == 75.0

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, cache_manager, mock_redis):
        """Test graceful error handling"""
        # Simulate Redis error
        mock_redis.get.side_effect = Exception("Redis connection error")

        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }

        # Should return None on error, not raise
        result = await cache_manager.get(request_data)
        assert result is None
