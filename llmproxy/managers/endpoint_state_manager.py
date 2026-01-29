from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Awaitable, Dict, List, Optional, Tuple, cast

import redis.asyncio as redis

from llmproxy.core.logger import get_logger
from llmproxy.core.redis_utils import await_redis_result
from llmproxy.models.endpoint import Endpoint

logger = get_logger(__name__)

_RECORD_OUTCOME_LUA = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local allowed_fails = tonumber(ARGV[2]) or 0
local cooldown_time = tonumber(ARGV[3]) or 0
local success_flag = tonumber(ARGV[4]) or 0
local error_message = ARGV[5]
local error_is_null = tonumber(ARGV[6]) or 0
local now_iso = ARGV[7]
local cooldown_until_iso = ARGV[8]
local default_state_json = ARGV[9]

local raw_state = redis.call('GET', key)
local state

if not raw_state then
    state = cjson.decode(default_state_json)
else
    local ok, decoded = pcall(cjson.decode, raw_state)
    if ok and type(decoded) == 'table' then
        state = decoded
    else
        state = cjson.decode(default_state_json)
    end
end

local function to_int(val)
    if type(val) == 'number' then
        return math.floor(val)
    elseif type(val) == 'string' then
        local num = tonumber(val)
        if num then
            return math.floor(num)
        end
    end
    return 0
end

state['allowed_fails'] = allowed_fails

local total_requests = to_int(state['total_requests']) + 1
local failed_requests = to_int(state['failed_requests'])
local consecutive_failures = to_int(state['consecutive_failures'])

state['total_requests'] = total_requests

if success_flag == 1 then
    state['consecutive_failures'] = 0
    state['status'] = 'healthy'
    state['cooldown_until'] = cjson.null
    state['last_success_time'] = now_iso
else
    failed_requests = failed_requests + 1
    consecutive_failures = consecutive_failures + 1

    state['failed_requests'] = failed_requests
    state['consecutive_failures'] = consecutive_failures

    if error_is_null == 1 then
        state['last_error'] = cjson.null
    else
        state['last_error'] = error_message
    end
    state['last_error_time'] = now_iso

    if consecutive_failures >= allowed_fails then
        state['status'] = 'cooling_down'
        state['cooldown_until'] = cooldown_until_iso
    end
end

state['updated_at'] = now_iso

local encoded = cjson.encode(state)

if ttl and ttl > 0 then
    redis.call('SETEX', key, ttl, encoded)
else
    redis.call('SET', key, encoded)
end

return encoded
"""

_MARK_HEALTHY_LUA = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local now_iso = ARGV[2]

local raw_state = redis.call('GET', key)
if not raw_state then
    return 0
end

local ok, state = pcall(cjson.decode, raw_state)
if not ok or type(state) ~= 'table' then
    return 0
end

state['status'] = 'healthy'
state['consecutive_failures'] = 0
state['cooldown_until'] = cjson.null
state['updated_at'] = now_iso

local encoded = cjson.encode(state)
if ttl and ttl > 0 then
    redis.call('SETEX', key, ttl, encoded)
else
    redis.call('SET', key, encoded)
end

return 1
"""


class EndpointStateManager:
    """Complete endpoint state management via Redis (single source of truth)"""

    def __init__(self, redis_client: redis.Redis, state_ttl: int = 3600):
        self.redis = redis_client
        self.state_ttl = state_ttl  # How long to keep endpoint state in Redis
        self._record_outcome_sha: Optional[str] = None
        self._mark_healthy_sha: Optional[str] = None

    async def _script_load(self, script: str) -> str:
        load_result = self.redis.script_load(script)
        if asyncio.isfuture(load_result) or asyncio.iscoroutine(load_result):
            return await cast(Awaitable[str], load_result)
        return cast(str, load_result)

    async def _evalsha(self, sha: str, keys: List[str], args: List[str]) -> Any:
        eval_result = self.redis.evalsha(sha, len(keys), *keys, *args)
        if asyncio.isfuture(eval_result) or asyncio.iscoroutine(eval_result):
            return await cast(Awaitable[Any], eval_result)
        return eval_result

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

    def _create_default_state(
        self, endpoint_id: str, allowed_fails: int
    ) -> Dict[str, Any]:
        """Create a default endpoint state"""
        return {
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

    def _safe_int_conversion(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int, returning default if conversion fails"""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    def _handle_success_outcome(self, state_data: Dict[str, Any]) -> None:
        """Handle successful request outcome"""
        state_data["consecutive_failures"] = 0
        state_data["status"] = "healthy"
        state_data["cooldown_until"] = None
        state_data["last_success_time"] = datetime.utcnow().isoformat()

    def _handle_failure_outcome(
        self,
        state_data: Dict[str, Any],
        error: Optional[str],
        allowed_fails: int,
        cooldown_time: int,
    ) -> None:
        """Handle failed request outcome"""
        failed_requests = state_data.get("failed_requests", 0)
        failed_requests_int = self._safe_int_conversion(failed_requests, 0)
        state_data["failed_requests"] = failed_requests_int + 1

        consecutive_failures = state_data.get("consecutive_failures", 0)
        consecutive_failures_int = self._safe_int_conversion(consecutive_failures, 0)
        state_data["consecutive_failures"] = consecutive_failures_int + 1

        state_data["last_error"] = error
        state_data["last_error_time"] = datetime.utcnow().isoformat()

        # Check if we should enter cooldown
        current_failures_val = state_data["consecutive_failures"]
        current_failures = self._safe_int_conversion(current_failures_val, 0)
        if current_failures >= allowed_fails:
            state_data["status"] = "cooling_down"
            cooldown_until = datetime.utcnow() + timedelta(seconds=cooldown_time)
            state_data["cooldown_until"] = cooldown_until.isoformat()

    async def record_request_outcome(
        self,
        endpoint_id: str,
        success: bool,
        error: Optional[str] = None,
        allowed_fails: int = 1,
        cooldown_time: int = 60,
    ) -> None:
        """Record a request outcome and update endpoint state atomically"""
        try:
            key = self._get_endpoint_key(endpoint_id)

            now = datetime.utcnow()
            now_iso = now.isoformat()
            cooldown_until_iso = (now + timedelta(seconds=cooldown_time)).isoformat()
            default_state_json = json.dumps(
                self._create_default_state(endpoint_id, allowed_fails)
            )

            sha = self._record_outcome_sha
            if not sha:
                sha = await self._script_load(_RECORD_OUTCOME_LUA)
                self._record_outcome_sha = sha

            error_is_null = 1 if error is None else 0

            string_args = [
                str(self.state_ttl),
                str(allowed_fails),
                str(cooldown_time),
                str(1 if success else 0),
                error or "",
                str(error_is_null),
                now_iso,
                cooldown_until_iso,
                default_state_json,
            ]

            try:
                updated_state_raw = await self._evalsha(
                    sha,
                    [key],
                    string_args,
                )
            except redis.ResponseError as exc:
                if "NOSCRIPT" in str(exc):
                    sha = await self._script_load(_RECORD_OUTCOME_LUA)
                    self._record_outcome_sha = sha
                    updated_state_raw = await self._evalsha(
                        sha,
                        [key],
                        string_args,
                    )
                else:
                    raise

            updated_state: Dict[str, Any] = {}
            if updated_state_raw:
                try:
                    updated_state = json.loads(updated_state_raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "endpoint_request_record_decode_failed",
                        endpoint_id=endpoint_id,
                        raw_state=updated_state_raw,
                    )

            logger.debug(
                "endpoint_request_recorded",
                endpoint_id=endpoint_id,
                success=success,
                total_requests=updated_state.get("total_requests"),
                failed_requests=updated_state.get("failed_requests"),
                status=updated_state.get("status"),
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
            now_iso = datetime.utcnow().isoformat()

            sha = self._mark_healthy_sha
            if not sha:
                sha = await self._script_load(_MARK_HEALTHY_LUA)
                self._mark_healthy_sha = sha

            try:
                await self._evalsha(sha, [key], [str(self.state_ttl), now_iso])
            except redis.ResponseError as exc:
                if "NOSCRIPT" in str(exc):
                    sha = await self._script_load(_MARK_HEALTHY_LUA)
                    self._mark_healthy_sha = sha
                    await self._evalsha(sha, [key], [str(self.state_ttl), now_iso])
                else:
                    raise

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

    async def get_endpoint_states_bulk(
        self, endpoint_ids: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Fetch multiple endpoint states with a single Redis request."""
        if not endpoint_ids:
            return {}

        keys = [self._get_endpoint_key(endpoint_id) for endpoint_id in endpoint_ids]
        states: Dict[str, Optional[Dict[str, Any]]] = {
            endpoint_id: None for endpoint_id in endpoint_ids
        }

        try:
            raw_results = await self.redis.mget(*keys)
        except Exception as exc:
            logger.error(
                "endpoint_states_bulk_fetch_failed",
                endpoint_ids=endpoint_ids,
                error=str(exc),
            )
            return states

        for endpoint_id, raw_state in zip(endpoint_ids, raw_results):
            if not raw_state:
                continue
            try:
                parsed = json.loads(raw_state)
                if isinstance(parsed, dict):
                    states[endpoint_id] = parsed
            except json.JSONDecodeError:
                logger.warning(
                    "endpoint_state_bulk_decode_failed",
                    endpoint_id=endpoint_id,
                )

        return states

    def _normalize_state_for_availability(
        self, endpoint_id: str, state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Dict[str, Any]], bool]:
        """Determine availability and whether cooldown cleanup is required."""
        if not state:
            return True, None, False

        status = state.get("status", "healthy")
        if status == "healthy":
            return True, state, False

        if status == "cooling_down":
            cooldown_until_str = state.get("cooldown_until")
            if cooldown_until_str:
                try:
                    cooldown_until = datetime.fromisoformat(cooldown_until_str)
                    if datetime.utcnow() > cooldown_until:
                        return True, state, True
                except ValueError:
                    logger.warning(
                        "endpoint_malformed_cooldown_timestamp",
                        endpoint_id=endpoint_id,
                        cooldown_until=cooldown_until_str,
                    )
                    return True, state, True
            return False, state, False

        return False, state, False

    async def get_availability_bulk(
        self, endpoint_ids: List[str]
    ) -> Tuple[Dict[str, bool], Dict[str, Optional[Dict[str, Any]]]]:
        """Compute availability for multiple endpoints in one Redis pass."""
        states = await self.get_endpoint_states_bulk(endpoint_ids)
        availability: Dict[str, bool] = {}
        cooldown_cleanup: List[str] = []

        for endpoint_id in endpoint_ids:
            state = states.get(endpoint_id)
            (
                is_available,
                state_data,
                needs_cleanup,
            ) = self._normalize_state_for_availability(endpoint_id, state)
            availability[endpoint_id] = is_available
            states[endpoint_id] = state_data
            if needs_cleanup:
                cooldown_cleanup.append(endpoint_id)

        if cooldown_cleanup:
            await asyncio.gather(
                *(
                    self._mark_endpoint_healthy(endpoint_id)
                    for endpoint_id in cooldown_cleanup
                )
            )

        return availability, states

    async def refresh_ttl_bulk(self, endpoint_ids: List[str]) -> None:
        """Refresh TTL for a batch of endpoint states."""
        if self.state_ttl <= 0 or not endpoint_ids:
            return

        async with self.redis.pipeline(transaction=False) as pipe:
            for endpoint_id in endpoint_ids:
                pipe.expire(self._get_endpoint_key(endpoint_id), self.state_ttl)
            await pipe.execute()

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

    async def get_endpoint_stats_bulk(
        self, endpoint_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        states = await self.get_endpoint_states_bulk(endpoint_ids)
        stats: Dict[str, Dict[str, Any]] = {}

        for endpoint_id in endpoint_ids:
            state_data = states.get(endpoint_id)
            if not state_data:
                stats[endpoint_id] = {
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
                continue

            total_requests = state_data.get("total_requests", 0)
            failed_requests = state_data.get("failed_requests", 0)
            success_rate = 0
            if total_requests > 0:
                success_rate = (
                    (total_requests - failed_requests) / total_requests
                ) * 100

            stats[endpoint_id] = {
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

        return stats

    async def register_endpoint_in_group(
        self, model_group: str, endpoint_id: str
    ) -> None:
        """Register an endpoint as part of a model group"""
        try:
            key = self._get_model_group_key(model_group)
            await await_redis_result(self.redis.sadd(key, endpoint_id))
            await await_redis_result(self.redis.expire(key, self.state_ttl))

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
            endpoint_ids = await await_redis_result(self.redis.smembers(key))
            return list(endpoint_ids) if endpoint_ids else []

        except Exception as e:
            logger.error(
                "group_endpoints_fetch_failed", model_group=model_group, error=str(e)
            )
            return []

    async def cleanup_stale_states(self, active_endpoint_ids: List[str]) -> None:
        """Clean up stale endpoint states that are no longer active"""
        try:
            pattern = "llmproxy:endpoint:*"
            active_ids_set = set(active_endpoint_ids)
            batch: List[str] = []
            batch_size = 256
            deleted_total = 0

            async for key in self.redis.scan_iter(match=pattern):
                endpoint_id = key.split(":")[-1]
                if endpoint_id in active_ids_set:
                    continue

                batch.append(key)
                if len(batch) >= batch_size:
                    deleted_total += await self._delete_keys_batch(batch)
                    batch = []

            if batch:
                deleted_total += await self._delete_keys_batch(batch)

            if deleted_total > 0:
                logger.info("stale_endpoint_states_cleaned", count=deleted_total)

        except Exception as e:
            logger.error("stale_state_cleanup_failed", error=str(e))

    async def _delete_keys_batch(self, keys: List[str]) -> int:
        if not keys:
            return 0

        try:
            if hasattr(self.redis, "unlink"):
                unlink_callable = getattr(self.redis, "unlink")
                unlink_result = unlink_callable(*keys)
                if asyncio.isfuture(unlink_result) or asyncio.iscoroutine(
                    unlink_result
                ):
                    deleted_result = await cast(Awaitable[Any], unlink_result)
                else:
                    deleted_result = unlink_result
            else:
                delete_result = self.redis.delete(*keys)
                if asyncio.isfuture(delete_result) or asyncio.iscoroutine(
                    delete_result
                ):
                    deleted_result = await cast(Awaitable[Any], delete_result)
                else:
                    deleted_result = delete_result

            return int(deleted_result)
        except Exception as exc:
            logger.warning("stale_state_batch_delete_failed", error=str(exc))
            return 0

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
