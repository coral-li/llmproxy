from typing import List, Optional, Dict
from datetime import datetime
import random
import asyncio
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.endpoint import Endpoint, EndpointStatus
from config_model import LLMProxyConfig, ModelGroup
from managers.rate_limit_manager import RateLimitManager
from core.logger import get_logger

logger = get_logger(__name__)


class LoadBalancer:
    """Manages endpoint pools and implements weighted round-robin with health checks"""

    def __init__(self, cooldown_time: int = 60, allowed_fails: int = 1):
        self.cooldown_time = cooldown_time
        self.allowed_fails = allowed_fails
        self.endpoint_pools: Dict[str, List[Endpoint]] = {}
        self._lock = asyncio.Lock()
        self.rate_limit_manager: Optional[RateLimitManager] = None

    def set_rate_limit_manager(self, manager: RateLimitManager):
        """Set the rate limit manager instance"""
        self.rate_limit_manager = manager

    async def initialize_from_config(self, config: LLMProxyConfig):
        """Initialize endpoint pools from configuration"""
        for model_group in config.model_groups:
            pool = []

            for model_config in model_group.models:
                endpoint = Endpoint(
                    model=model_config.model,
                    weight=model_config.weight,
                    params=model_config.params,
                    allowed_fails=config.general_settings.allowed_fails,
                )
                pool.append(endpoint)

                logger.info(
                    "endpoint_added",
                    model_group=model_group.model_group,
                    model=model_config.model,
                    weight=model_config.weight,
                    base_url=endpoint.base_url,
                )

            self.endpoint_pools[model_group.model_group] = pool

        logger.info(
            "load_balancer_initialized",
            model_groups=list(self.endpoint_pools.keys()),
            total_endpoints=sum(len(pool) for pool in self.endpoint_pools.values()),
        )

    async def select_endpoint(self, model_group: str) -> Optional[Endpoint]:
        """Select an endpoint using weighted round-robin with health checks"""
        async with self._lock:
            pool = self.endpoint_pools.get(model_group, [])

            if not pool:
                logger.error("no_endpoints_configured", model_group=model_group)
                return None

            # Filter available endpoints
            available = []
            for ep in pool:
                if not ep.is_available():
                    continue

                # Check rate limits if manager is available
                if self.rate_limit_manager:
                    (
                        is_available,
                        wait_time,
                    ) = await self.rate_limit_manager.check_availability(ep.id)
                    if not is_available:
                        logger.debug(
                            "endpoint_rate_limited",
                            endpoint_id=ep.id,
                            wait_time=wait_time,
                        )
                        continue

                available.append(ep)

            if not available:
                # All endpoints are down or rate limited, try weight=0 fallbacks
                logger.warning(
                    "all_primary_endpoints_unavailable", model_group=model_group
                )

                fallbacks = [ep for ep in pool if ep.weight == 0 and ep.is_available()]
                if self.rate_limit_manager:
                    # Check rate limits for fallbacks too
                    available_fallbacks = []
                    for ep in fallbacks:
                        (
                            is_available,
                            _,
                        ) = await self.rate_limit_manager.check_availability(ep.id)
                        if is_available:
                            available_fallbacks.append(ep)
                    available = available_fallbacks
                else:
                    available = fallbacks

            if not available:
                logger.error("no_available_endpoints", model_group=model_group)
                return None

            # Weighted random selection
            total_weight = sum(ep.weight for ep in available)

            # If all weights are 0 (all fallbacks), do uniform random
            if total_weight == 0:
                selected = random.choice(available)
            else:
                # Weighted random selection
                rand = random.uniform(0, total_weight)
                cumulative = 0

                for endpoint in available:
                    cumulative += endpoint.weight
                    if rand <= cumulative:
                        selected = endpoint
                        break
                else:
                    selected = available[-1]

            logger.info(
                "endpoint_selected",
                model_group=model_group,
                endpoint_id=selected.id,
                model=selected.model,
                base_url=selected.base_url,
                weight=selected.weight,
            )

            return selected

    def record_success(self, endpoint: Endpoint):
        """Record successful request for an endpoint"""
        endpoint.record_success()
        logger.debug(
            "endpoint_success",
            endpoint_id=endpoint.id,
            consecutive_failures=endpoint.consecutive_failures,
        )

    def record_failure(self, endpoint: Endpoint, error: str):
        """Record failed request for an endpoint"""
        endpoint.record_failure(error, self.cooldown_time)
        logger.warning(
            "endpoint_failure",
            endpoint_id=endpoint.id,
            error=error,
            consecutive_failures=endpoint.consecutive_failures,
            status=endpoint.status.value,
        )

    def get_all_endpoints_stats(self) -> Dict[str, List[Dict]]:
        """Get statistics for all endpoints"""
        stats = {}
        for model_group, endpoints in self.endpoint_pools.items():
            stats[model_group] = [ep.get_stats() for ep in endpoints]
        return stats

    def get_model_groups(self) -> List[str]:
        """Get list of configured model groups"""
        return list(self.endpoint_pools.keys())
