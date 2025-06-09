import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from llmproxy.core.logger import get_logger
from llmproxy.models.endpoint import Endpoint

logger = get_logger(__name__)


class EndpointStateManager:
    """Complete endpoint state management via Redis (single source of truth)"""

    def __init__(self, redis_client: redis.Redis, state_ttl: int = 3600):
        self.redis = redis_client
        self.state_ttl = state_ttl  # How long to keep endpoint state in Redis
        self._lock = asyncio.Lock()

    def _get_endpoint_key(self, endpoint_id: str) -> str:
        """Get Redis key for endpoint state"""
        return f"llmproxy:endpoint:{endpoint_id}"

    def _get_model_group_key(self, model_group: str) -> str:
        """Get Redis key for model group endpoint list"""
        return f"llmproxy:model_group:{model_group}"

    async def initialize_endpoint(self, endpoint: Endpoint, model_group: str) -> None:
        """Initialize endpoint state in Redis if it doesn't exist"""
        try:
            key = self._get_endpoint_key(endpoint.id)

            # Check if state already exists
            existing_data = await self.redis.get(key)
            if not existing_data:
                # Initialize with default healthy state
                initial_state = {
                    "id": endpoint.id,
                    "model": endpoint.model,
                    "weight": endpoint.weight,
                    "base_url": endpoint.base_url,
                    "is_azure": endpoint.is_azure,
                    "allowed_fails": endpoint.allowed_fails,
                    "status": "healthy",
                    "total_requests": 0,
                    "failed_requests": 0,
                    "consecutive_failures": 0,
                    "last_error": None,
                    "last_error_time": None,
                    "last_success_time": None,
                    "cooldown_until": None,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }

                await self.redis.setex(key, self.state_ttl, json.dumps(initial_state))
                await self.register_endpoint_in_group(model_group, endpoint.id)

                logger.info(
                    "endpoint_state_initialized",
                    endpoint_id=endpoint.id,
                    model_group=model_group,
                    base_url=endpoint.base_url,
                )
            else:
                logger.info(
                    "endpoint_state_exists",
                    endpoint_id=endpoint.id,
                    model_group=model_group,
                )

        except Exception as e:
            logger.error(
                "endpoint_state_init_failed", endpoint_id=endpoint.id, error=str(e)
            )

    async def record_request_outcome(
        self,
        endpoint_id: str,
        success: bool,
        error: Optional[str] = None,
        allowed_fails: int = 1,
        cooldown_time: int = 60,
    ) -> None:
        """Record a request outcome and update endpoint state atomically"""  # noqa: C901
        try:
            key = self._get_endpoint_key(endpoint_id)

            # Get current state
            current_data = await self.redis.get(key)
            if not current_data:
                logger.info(
                    "endpoint_state_missing_for_request_creating_default",
                    endpoint_id=endpoint_id,
                )
                # Create a minimal default state if endpoint state is missing
                state_data = {
                    "id": endpoint_id,
                    "model": "unknown",
                    "weight": 1,
                    "base_url": "unknown",
                    "is_azure": False,
                    "allowed_fails": allowed_fails,
                    "status": "healthy",
                    "total_requests": 0,
                    "failed_requests": 0,
                    "consecutive_failures": 0,
                    "last_error": None,
                    "last_error_time": None,
                    "last_success_time": None,
                    "cooldown_until": None,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            else:
                try:
                    state_data = json.loads(current_data)
                except json.JSONDecodeError:
                    logger.warning(
                        "endpoint_state_corrupted_creating_default",
                        endpoint_id=endpoint_id,
                    )
                    # Create default state if JSON is corrupted
                    state_data = {
                        "id": endpoint_id,
                        "model": "unknown",
                        "weight": 1,
                        "base_url": "unknown",
                        "is_azure": False,
                        "allowed_fails": allowed_fails,
                        "status": "healthy",
                        "total_requests": 0,
                        "failed_requests": 0,
                        "consecutive_failures": 0,
                        "last_error": None,
                        "last_error_time": None,
                        "last_success_time": None,
                        "cooldown_until": None,
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                    }

            # Update counters with safe conversion
            def safe_int_conversion(value: Any, default: int = 0) -> int:
                """Safely convert value to int, returning default if conversion fails"""
                if isinstance(value, int):
                    return value
                if isinstance(value, str):
                    try:
                        return int(value)
                    except ValueError:
                        return default
                return default

            total_requests = state_data.get("total_requests", 0)
            total_requests_int = safe_int_conversion(total_requests, 0)
            state_data["total_requests"] = total_requests_int + 1

            if success:
                # Success: reset failures, mark healthy
                state_data["consecutive_failures"] = 0
                state_data["status"] = "healthy"
                state_data["cooldown_until"] = None
                state_data["last_success_time"] = datetime.utcnow().isoformat()
            else:
                # Failure: increment counters, check for cooldown
                failed_requests = state_data.get("failed_requests", 0)
                failed_requests_int = safe_int_conversion(failed_requests, 0)
                state_data["failed_requests"] = failed_requests_int + 1

                consecutive_failures = state_data.get("consecutive_failures", 0)
                consecutive_failures_int = safe_int_conversion(consecutive_failures, 0)
                state_data["consecutive_failures"] = consecutive_failures_int + 1

                state_data["last_error"] = error
                state_data["last_error_time"] = datetime.utcnow().isoformat()

                # Check if we should enter cooldown
                current_failures_val = state_data["consecutive_failures"]
                current_failures = safe_int_conversion(current_failures_val, 0)
                if current_failures >= allowed_fails:
                    state_data["status"] = "cooling_down"
                    cooldown_until = datetime.utcnow() + timedelta(
                        seconds=cooldown_time
                    )
                    state_data["cooldown_until"] = cooldown_until.isoformat()

            state_data["updated_at"] = datetime.utcnow().isoformat()

            # Save updated state atomically
            await self.redis.setex(key, self.state_ttl, json.dumps(state_data))

            logger.debug(
                "endpoint_request_recorded",
                endpoint_id=endpoint_id,
                success=success,
                total_requests=state_data["total_requests"],
                failed_requests=state_data["failed_requests"],
                status=state_data["status"],
            )

        except Exception as e:
            logger.error(
                "endpoint_request_record_failed",
                endpoint_id=endpoint_id,
                success=success,
                error=str(e),
            )

    async def is_endpoint_available(self, endpoint_id: str) -> bool:
        """Check if endpoint is available for requests (from Redis)"""
        try:
            state_data = await self.get_endpoint_state(endpoint_id)
            if not state_data:
                return True  # Default to available if no state found

            status = state_data.get("status", "healthy")

            if status == "healthy":
                return True
            elif status == "cooling_down":
                cooldown_until_str = state_data.get("cooldown_until")
                if cooldown_until_str:
                    try:
                        cooldown_until = datetime.fromisoformat(cooldown_until_str)
                        if datetime.utcnow() > cooldown_until:
                            # Cooldown expired, mark as healthy
                            await self._mark_endpoint_healthy(endpoint_id)
                            return True
                    except ValueError:
                        logger.warning(
                            "endpoint_malformed_cooldown_timestamp",
                            endpoint_id=endpoint_id,
                            cooldown_until=cooldown_until_str,
                        )
                        # Treat malformed timestamp as expired cooldown
                        await self._mark_endpoint_healthy(endpoint_id)
                        return True
                return False

            return False

        except Exception as e:
            logger.error(
                "endpoint_availability_check_failed",
                endpoint_id=endpoint_id,
                error=str(e),
            )
            return True  # Default to available on error

    async def _mark_endpoint_healthy(self, endpoint_id: str) -> None:
        """Mark endpoint as healthy when cooldown expires"""
        try:
            key = self._get_endpoint_key(endpoint_id)
            current_data = await self.redis.get(key)
            if current_data:
                state_data = json.loads(current_data)
                state_data["status"] = "healthy"
                state_data["consecutive_failures"] = 0
                state_data["cooldown_until"] = None
                state_data["updated_at"] = datetime.utcnow().isoformat()

                await self.redis.setex(key, self.state_ttl, json.dumps(state_data))

                logger.debug("endpoint_marked_healthy", endpoint_id=endpoint_id)

        except Exception as e:
            logger.error(
                "endpoint_health_marking_failed", endpoint_id=endpoint_id, error=str(e)
            )

    async def get_endpoint_state(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get complete endpoint state from Redis"""
        try:
            key = self._get_endpoint_key(endpoint_id)
            data = await self.redis.get(key)

            if data:
                parsed_data = json.loads(data)
                return parsed_data if isinstance(parsed_data, dict) else None
            return None

        except Exception as e:
            logger.error(
                "endpoint_state_get_failed", endpoint_id=endpoint_id, error=str(e)
            )
            return None

    async def get_endpoint_stats(self, endpoint_id: str) -> Dict[str, Any]:
        """Get endpoint statistics with calculated success rate"""
        state_data = await self.get_endpoint_state(endpoint_id)
        if not state_data:
            return {
                "id": endpoint_id,
                "status": "unknown",
                "total_requests": 0,
                "failed_requests": 0,
                "success_rate": 0,
                "consecutive_failures": 0,
                "last_error": None,
                "last_error_time": None,
                "last_success_time": None,
                "cooldown_until": None,
            }

        total_requests = state_data.get("total_requests", 0)
        failed_requests = state_data.get("failed_requests", 0)
        success_rate = 0
        if total_requests > 0:
            success_rate = ((total_requests - failed_requests) / total_requests) * 100

        return {
            "id": state_data.get("id", endpoint_id),
            "model": state_data.get("model"),
            "weight": state_data.get("weight"),
            "status": state_data.get("status", "unknown"),
            "base_url": state_data.get("base_url"),
            "is_azure": state_data.get("is_azure"),
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "consecutive_failures": state_data.get("consecutive_failures", 0),
            "last_error": state_data.get("last_error"),
            "last_error_time": state_data.get("last_error_time"),
            "last_success_time": state_data.get("last_success_time"),
            "cooldown_until": state_data.get("cooldown_until"),
        }

    async def register_endpoint_in_group(
        self, model_group: str, endpoint_id: str
    ) -> None:
        """Register an endpoint as part of a model group"""
        try:
            key = self._get_model_group_key(model_group)
            await self.redis.sadd(key, endpoint_id)  # type: ignore
            await self.redis.expire(key, self.state_ttl)

            logger.debug(
                "endpoint_registered_in_group",
                model_group=model_group,
                endpoint_id=endpoint_id,
            )

        except Exception as e:
            logger.error(
                "endpoint_group_registration_failed",
                model_group=model_group,
                endpoint_id=endpoint_id,
                error=str(e),
            )

    async def get_group_endpoints(self, model_group: str) -> List[str]:
        """Get all endpoint IDs for a model group"""
        try:
            key = self._get_model_group_key(model_group)
            endpoint_ids = await self.redis.smembers(key)  # type: ignore
            return list(endpoint_ids) if endpoint_ids else []

        except Exception as e:
            logger.error(
                "group_endpoints_fetch_failed", model_group=model_group, error=str(e)
            )
            return []

    async def cleanup_stale_states(self, active_endpoint_ids: List[str]) -> None:
        """Clean up stale endpoint states that are no longer active"""
        try:
            # Find all endpoint keys
            pattern = "llmproxy:endpoint:*"
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            # Identify stale keys
            stale_keys = []
            for key in keys:
                endpoint_id = key.split(":")[-1]
                if endpoint_id not in active_endpoint_ids:
                    stale_keys.append(key)

            # Delete stale keys
            if stale_keys:
                await self.redis.delete(*stale_keys)
                logger.info("stale_endpoint_states_cleaned", count=len(stale_keys))

        except Exception as e:
            logger.error("stale_state_cleanup_failed", error=str(e))

    async def health_check(self) -> bool:
        """Check if Redis state management is working"""
        try:
            test_key = "llmproxy:health_check"
            test_value = datetime.utcnow().isoformat()

            await self.redis.setex(test_key, 60, test_value)
            result = await self.redis.get(test_key)
            await self.redis.delete(test_key)

            return bool(result == test_value)

        except Exception as e:
            logger.error("state_manager_health_check_failed", error=str(e))
            return False

    async def ensure_endpoint_state(self, endpoint: Endpoint, model_group: str) -> None:
        """Ensure endpoint has proper state, creating or updating as needed"""
        try:
            key = self._get_endpoint_key(endpoint.id)

            # Get current state
            current_data = await self.redis.get(key)
            if not current_data:
                # No state exists, initialize fully
                await self.initialize_endpoint(endpoint, model_group)
                return

            state_data = json.loads(current_data)

            # Check if this is a minimal default state (missing proper endpoint info)
            if (
                state_data.get("model") == "unknown"
                or state_data.get("base_url") == "unknown"
            ):
                logger.info("upgrading_minimal_endpoint_state", endpoint_id=endpoint.id)

                # Preserve counters and status but update endpoint info
                state_data.update(
                    {
                        "model": endpoint.model,
                        "weight": endpoint.weight,
                        "base_url": endpoint.base_url,
                        "is_azure": endpoint.is_azure,
                        "allowed_fails": endpoint.allowed_fails,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )

                # Save updated state
                await self.redis.setex(key, self.state_ttl, json.dumps(state_data))
                await self.register_endpoint_in_group(model_group, endpoint.id)

                logger.info(
                    "endpoint_state_upgraded",
                    endpoint_id=endpoint.id,
                    model_group=model_group,
                )

        except Exception as e:
            logger.error(
                "endpoint_state_ensure_failed", endpoint_id=endpoint.id, error=str(e)
            )
