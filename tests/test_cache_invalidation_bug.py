import unittest.mock as mock

import pytest

from llmproxy.core.cache_manager import CacheManager


@pytest.mark.asyncio
async def test_invalidate_request_handles_key_generation_error():
    """Test that invalidate_request handles errors in key generation gracefully"""

    # Mock Redis client
    mock_redis = mock.AsyncMock()

    # Create cache manager
    cache_manager = CacheManager(mock_redis)

    # Mock _generate_cache_key to raise an exception
    with mock.patch.object(
        cache_manager,
        "_generate_cache_key",
        side_effect=Exception("Key generation failed"),
    ):
        # This should not raise a NameError
        result = await cache_manager.invalidate_request({"model": "test"})

        # Should return False on error
        assert result is False

    # Verify that no Redis operations were called since key generation failed
    mock_redis.exists.assert_not_called()
    mock_redis.delete.assert_not_called()


@pytest.mark.asyncio
async def test_invalidate_request_works_normally():
    """Test that invalidate_request still works normally when no errors occur"""

    # Mock Redis client
    mock_redis = mock.AsyncMock()
    mock_redis.exists.return_value = True
    mock_redis.delete.return_value = 1

    # Create cache manager
    cache_manager = CacheManager(mock_redis)

    # Normal request should work
    result = await cache_manager.invalidate_request(
        {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    )

    # Should return True on success
    assert result is True

    # Should have checked for existence of cache keys
    assert (
        mock_redis.exists.call_count == 4
    )  # regular, stream, responses_stream, responses_normalized

    # Should have deleted existing keys
    assert mock_redis.delete.call_count == 4


@pytest.mark.asyncio
async def test_invalidate_request_handles_redis_error():
    """Test that invalidate_request handles Redis errors gracefully"""

    # Mock Redis client that raises error on exists()
    mock_redis = mock.AsyncMock()
    mock_redis.exists.side_effect = Exception("Redis connection failed")

    # Create cache manager
    cache_manager = CacheManager(mock_redis)

    # This should not raise an exception
    result = await cache_manager.invalidate_request(
        {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    )

    # Should return False on error
    assert result is False
