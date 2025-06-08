from typing import List, Optional, Dict
from datetime import datetime
import random
import asyncio
from pathlib import Path
import sys

# Imports are handled properly through package structure

from llmproxy.models.endpoint import Endpoint, EndpointStatus
from llmproxy.config_model import LLMProxyConfig, ModelGroup
from llmproxy.core.logger import get_logger

logger = get_logger(__name__)


class LoadBalancer:
    """Manages endpoint pools and implements weighted round-robin with health checks"""

    def __init__(self, cooldown_time: int = 60, allowed_fails: int = 1):
        self.cooldown_time = cooldown_time
        self.allowed_fails = allowed_fails
        self.endpoint_pools: Dict[str, List[Endpoint]] = {}
        self._lock = asyncio.Lock()

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

            # Log current state of all endpoints
            logger.info(
                "selecting_endpoint",
                model_group=model_group,
                endpoints=[
                    {
                        "id": ep.id,
                        "base_url": ep.base_url,
                        "weight": ep.weight,
                        "status": ep.status.value,
                        "is_available": ep.is_available(),
                        "consecutive_failures": ep.consecutive_failures,
                    }
                    for ep in pool
                ]
            )

            # First, try endpoints with weight > 0
            primary_available = []
            for ep in pool:
                if ep.weight == 0:  # Skip weight=0 endpoints in primary pass
                    continue
                    
                if not ep.is_available():
                    continue

                primary_available.append(ep)

            # Use primary endpoints if available
            if primary_available:
                available = primary_available
            else:
                # All primary endpoints are down, try weight=0 fallbacks
                logger.warning(
                    "all_primary_endpoints_unavailable", model_group=model_group
                )

                fallbacks = [ep for ep in pool if ep.weight == 0 and ep.is_available()]
                logger.info(
                    "fallback_endpoints_found",
                    model_group=model_group,
                    num_fallbacks=len(fallbacks),
                    fallback_ids=[ep.id for ep in fallbacks],
                    fallback_base_urls=[ep.base_url for ep in fallbacks],
                    fallback_statuses=[ep.status.value for ep in fallbacks]
                )
                
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
