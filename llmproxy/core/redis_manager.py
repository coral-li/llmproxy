from typing import Optional, Union

import redis.asyncio as redis

from .logger import get_logger

logger = get_logger(__name__)


class RedisManager:
    """Manages Redis connections with health checks and retry logic"""

    def __init__(
        self, host: str, port: Union[int, str], password: Optional[str] = None
    ):
        self.host = host
        self.port = int(port)
        self.password = password
        self.client: Optional[redis.Redis] = None
        self._pool: Optional[redis.ConnectionPool] = None

    async def connect(self) -> None:
        """Initialize Redis connection with connection pooling"""
        try:
            self._pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
                max_connections=50,
            )

            self.client = redis.Redis(connection_pool=self._pool)

            # Test connection
            assert self.client is not None  # mypy: Redis constructor cannot return None
            await self.client.ping()
            logger.info("redis_connected", host=self.host, port=self.port)

        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self.client:
            # Close Redis client (handles both newer aclose() and older close() methods)
            if hasattr(self.client, "aclose"):
                await self.client.aclose()
            else:
                await self.client.close()
            if self._pool:
                await self._pool.disconnect()
            logger.info("redis_disconnected")

    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            if self.client is not None:
                await self.client.ping()
                return True
            return False
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            return False

    def get_client(self) -> redis.Redis:
        """Get Redis client instance"""
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        return self.client
