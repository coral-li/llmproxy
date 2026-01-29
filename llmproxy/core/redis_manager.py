import inspect
from typing import Any, Awaitable, Optional, Union

import redis.asyncio as redis

from .logger import get_logger

logger = get_logger(__name__)


def _ensure_awaitable(value: object, error_label: str) -> Awaitable[Any]:
    if not inspect.isawaitable(value):
        raise TypeError(error_label)
    return value


class RedisManager:
    """Manages Redis connections with health checks and retry logic"""

    def __init__(
        self,
        host: str,
        port: Union[int, str],
        password: Optional[str] = None,
        ssl_enabled: bool = False,
        ssl_cert_reqs: Optional[str] = None,
    ):
        self.host = host
        self.port = int(port)
        self.password = password
        self.ssl_enabled = ssl_enabled
        self.ssl_cert_reqs = ssl_cert_reqs
        self.client: Optional[redis.Redis] = None
        self._pool: Optional[redis.ConnectionPool] = None

    async def connect(self) -> None:
        """Initialize Redis connection with connection pooling"""
        try:
            # Initialize the connection pool based on SSL usage
            if self.ssl_enabled:
                # Create SSL connection pool
                ssl_cert_reqs = (
                    self.ssl_cert_reqs.lower() if self.ssl_cert_reqs else None
                )
                self._pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    connection_class=redis.SSLConnection,
                    ssl_cert_reqs=ssl_cert_reqs,
                    decode_responses=True,
                    max_connections=50,
                )
            else:
                # Create standard connection pool
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
            await _ensure_awaitable(
                self.client.ping(), "redis_client_ping_not_awaitable"
            )
            logger.info(
                "redis_connected", host=self.host, port=self.port, ssl=self.ssl_enabled
            )

        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self.client:
            # Why check isawaitable: redis-py has varied sync/async APIs across versions.
            # We fail fast if these are not coroutines to avoid silently skipping cleanup.
            await _ensure_awaitable(
                self.client.aclose(), "redis_client_aclose_not_awaitable"
            )
            if self._pool:
                # Same rationale as above: assert awaitable to catch version drift early.
                await _ensure_awaitable(
                    self._pool.disconnect(), "redis_pool_disconnect_not_awaitable"
                )
            logger.info("redis_disconnected")

    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            if self.client is not None:
                await _ensure_awaitable(
                    self.client.ping(), "redis_client_ping_not_awaitable"
                )
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
