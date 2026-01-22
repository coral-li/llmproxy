from typing import Optional, cast

import redis.asyncio as redis

from llmproxy.core.logger import get_logger

logger = get_logger(__name__)


class ResponseAffinityManager:
    """Tracks response_id -> endpoint_id mappings for Responses API affinity."""

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 21600,
        namespace: str = "llmproxy:response_affinity",
    ) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    def _key(self, response_id: str) -> str:
        return f"{self.namespace}:{response_id}"

    async def get_endpoint_id(self, response_id: str) -> Optional[str]:
        if not response_id:
            return None
        key = self._key(response_id)
        try:
            return cast(Optional[str], await self.redis.get(key))
        except Exception as exc:  # pragma: no cover - Redis failures should not crash
            logger.error(
                "response_affinity_get_failed",
                response_id=response_id,
                error=str(exc),
            )
            return None

    async def set_endpoint_id(self, response_id: str, endpoint_id: str) -> None:
        if not response_id or not endpoint_id:
            return
        key = self._key(response_id)
        try:
            if self.ttl_seconds and self.ttl_seconds > 0:
                await self.redis.setex(key, self.ttl_seconds, endpoint_id)
            else:
                await self.redis.set(key, endpoint_id)
        except Exception as exc:  # pragma: no cover - Redis failures should not crash
            logger.error(
                "response_affinity_set_failed",
                response_id=response_id,
                endpoint_id=endpoint_id,
                error=str(exc),
            )
