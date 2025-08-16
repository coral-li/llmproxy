import asyncio
import random
from typing import Dict, List, Optional, Tuple

from llmproxy.config_model import LLMProxyConfig
from llmproxy.core.logger import get_logger
from llmproxy.managers.endpoint_state_manager import EndpointStateManager
from llmproxy.models.endpoint import Endpoint

# Imports are handled properly through package structure


logger = get_logger(__name__)


class LoadBalancer:
    """Stateless load balancer that uses Redis as single source of truth"""

    def __init__(
        self,
        cooldown_time: int = 60,
        allowed_fails: int = 1,
        state_manager: Optional[EndpointStateManager] = None,
    ):
        self.cooldown_time = cooldown_time
        self.allowed_fails = allowed_fails
        self.endpoint_configs: Dict[str, List[Endpoint]] = {}  # Just config, no state
        self.state_manager = state_manager
        # No internal lock needed; endpoint configurations are initialized once and treated as read-only

    def set_state_manager(self, state_manager: EndpointStateManager) -> None:
        """Set the endpoint state manager after initialization"""
        self.state_manager = state_manager

    async def initialize_from_config(self, config: LLMProxyConfig) -> None:
        """Initialize endpoint configurations and set up Redis state"""
        for model_group in config.model_groups:
            configs = []

            for model_config in model_group.models:
                # Create endpoint config (stateless)
                endpoint = Endpoint(
                    model=model_config.model,
                    weight=model_config.weight,
                    params=model_config.params,
                    allowed_fails=config.general_settings.allowed_fails,
                )

                # Initialize state in Redis
                if self.state_manager:
                    await self.state_manager.initialize_endpoint(
                        endpoint, model_group.model_group
                    )

                configs.append(endpoint)

                logger.info(
                    "endpoint_configured",
                    model_group=model_group.model_group,
                    model=model_config.model,
                    endpoint_id=endpoint.id,
                    weight=model_config.weight,
                    base_url=endpoint.base_url,
                )

            self.endpoint_configs[model_group.model_group] = configs

        logger.info(
            "load_balancer_initialized",
            model_groups=list(self.endpoint_configs.keys()),
            total_endpoints=sum(len(pool) for pool in self.endpoint_configs.values()),
        )

    async def select_endpoint(self, model_group: str) -> Optional[Endpoint]:
        """Select an endpoint using weighted round-robin with health checks from Redis"""
        # Read-only access to endpoint configs
        configs = list(self.endpoint_configs.get(model_group, []))

        if not configs:
            logger.error("no_endpoints_configured", model_group=model_group)
            return None

        if self.state_manager:
            await self._log_endpoint_states(configs, model_group)

        primary_available, fallback_available = await self._get_available_endpoints(
            configs
        )

        # Use primary endpoints if available
        if primary_available:
            available_endpoints = primary_available
        else:
            # All primary endpoints are down, try weight=0 fallbacks
            logger.warning("all_primary_endpoints_unavailable", model_group=model_group)
            available_endpoints = fallback_available

        if not available_endpoints:
            logger.error("no_available_endpoints", model_group=model_group)
            return None

        # Weighted random selection
        total_weight = sum(ep.weight for ep in available_endpoints)

        # If all weights are 0 (all fallbacks), do uniform random
        if total_weight == 0:
            selected = random.choice(available_endpoints)
        else:
            rand = random.uniform(0, total_weight)
            cumulative = 0

            for endpoint in available_endpoints:
                cumulative += endpoint.weight
                if rand <= cumulative:
                    selected = endpoint
                    break
            else:
                selected = available_endpoints[-1]

        logger.info(
            "endpoint_selected",
            model_group=model_group,
            endpoint_id=selected.id,
            model=selected.model,
            base_url=selected.base_url,
            weight=selected.weight,
        )

        return selected

    async def _get_available_endpoints(
        self, configs: List[Endpoint]
    ) -> Tuple[List[Endpoint], List[Endpoint]]:
        primary_available: List[Endpoint] = []
        fallback_available: List[Endpoint] = []

        if not configs:
            return primary_available, fallback_available

        if self.state_manager:
            availability_list = await asyncio.gather(
                *(
                    self.state_manager.is_endpoint_available(endpoint_config.id)
                    for endpoint_config in configs
                )
            )
        else:
            availability_list = [True for _ in configs]

        for endpoint_config, is_available in zip(configs, availability_list):
            if is_available:
                if endpoint_config.weight > 0:
                    primary_available.append(endpoint_config)
                else:
                    fallback_available.append(endpoint_config)

        return primary_available, fallback_available

    async def _log_endpoint_states(
        self, configs: List[Endpoint], model_group: str
    ) -> None:
        if not self.state_manager:
            logger.info(
                "selecting_endpoint",
                model_group=model_group,
                endpoints=[],
            )
            return

        # Fetch state and availability for all endpoints in parallel
        per_endpoint_results = await asyncio.gather(
            *(
                asyncio.gather(
                    self.state_manager.get_endpoint_state(config.id),
                    self.state_manager.is_endpoint_available(config.id),
                )
                for config in configs
            )
        )

        endpoint_states: List[Dict] = []
        for config, (state, is_available) in zip(configs, per_endpoint_results):
            endpoint_states.append(
                {
                    "id": config.id,
                    "base_url": config.base_url,
                    "weight": config.weight,
                    "status": (state.get("status", "unknown") if state else "unknown"),
                    "is_available": is_available,
                    "consecutive_failures": (
                        state.get("consecutive_failures", 0) if state else 0
                    ),
                }
            )

        logger.info(
            "selecting_endpoint",
            model_group=model_group,
            endpoints=endpoint_states,
        )

    async def record_success(self, endpoint: Endpoint) -> None:
        """Record successful request for an endpoint (in Redis)"""
        logger.debug(
            "endpoint_success",
            endpoint_id=endpoint.id,
        )

        # Ensure endpoint has proper state first
        if self.state_manager:
            await self.state_manager.ensure_endpoint_state(
                endpoint, self._get_model_group_for_endpoint(endpoint)
            )
            await self.state_manager.record_request_outcome(
                endpoint.id,
                success=True,
                allowed_fails=self.allowed_fails,
                cooldown_time=self.cooldown_time,
            )

    async def record_failure(self, endpoint: Endpoint, error: str) -> None:
        """Record failed request for an endpoint (in Redis)"""
        logger.warning(
            "endpoint_failure",
            endpoint_id=endpoint.id,
            error=error,
        )

        # Ensure endpoint has proper state first
        if self.state_manager:
            await self.state_manager.ensure_endpoint_state(
                endpoint, self._get_model_group_for_endpoint(endpoint)
            )
            await self.state_manager.record_request_outcome(
                endpoint.id,
                success=False,
                error=error,
                allowed_fails=self.allowed_fails,
                cooldown_time=self.cooldown_time,
            )

    def _get_model_group_for_endpoint(self, endpoint: Endpoint) -> str:
        """Find the model group for a given endpoint"""
        for model_group, configs in self.endpoint_configs.items():
            for config in configs:
                if config.id == endpoint.id:
                    return model_group
        return "unknown"

    async def get_all_endpoints_stats(self) -> Dict[str, List[Dict]]:
        """Get statistics for all endpoints (directly from Redis)"""
        stats = {}

        if not self.state_manager:
            # Return basic config if no state manager
            for model_group, configs in self.endpoint_configs.items():
                stats[model_group] = [config.get_config_dict() for config in configs]
            return stats

        for model_group, configs in self.endpoint_configs.items():
            endpoint_stats = []
            for config in configs:
                # Get fresh stats from Redis
                stat_data = await self.state_manager.get_endpoint_stats(config.id)
                endpoint_stats.append(stat_data)
            stats[model_group] = endpoint_stats

        return stats

    def get_model_groups(self) -> List[str]:
        """Get list of configured model groups"""
        return list(self.endpoint_configs.keys())

    async def shutdown(self) -> None:
        """Clean shutdown"""
        logger.info("load_balancer_shutdown_complete")
