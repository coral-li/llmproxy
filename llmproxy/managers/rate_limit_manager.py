import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import redis.asyncio as redis
import json
import re
from llmproxy.core.logger import get_logger

logger = get_logger(__name__)


class RateLimitManager:
    """Manages rate limits for endpoints using Redis for persistence"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._local_cache: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    def _parse_duration_string(self, duration_str: str) -> timedelta:
        """
        Parse OpenAI's duration string format into a timedelta object.
        Examples: '12ms', '4m12.172s', '23h18m29.144s'
        """
        if not duration_str:
            return timedelta(0)
        
        total_seconds = 0.0
        
        # Check if it's just milliseconds
        ms_match = re.match(r'^(\d+(?:\.\d+)?)ms$', duration_str)
        if ms_match:
            total_seconds = float(ms_match.group(1)) / 1000
            return timedelta(seconds=total_seconds)
        
        # Otherwise parse hours, minutes, seconds
        # Pattern to match components like 23h18m29.144s
        pattern = r'(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?'
        match = re.fullmatch(pattern, duration_str)
        
        if not match or not any(match.groups()):
            logger.warning(f"Could not parse duration string: {duration_str}")
            return timedelta(seconds=60)  # Default to 60 seconds
        
        hours, minutes, seconds = match.groups()
        
        if hours:
            total_seconds += float(hours) * 3600
        if minutes:
            total_seconds += float(minutes) * 60
        if seconds:
            total_seconds += float(seconds)
        
        return timedelta(seconds=total_seconds)

    async def check_availability(self, endpoint_id: str) -> Tuple[bool, Optional[int]]:
        """
        Check if endpoint has available capacity
        Returns: (is_available, seconds_until_reset)
        """
        key = f"ratelimit:{endpoint_id}"

        try:
            # Get rate limit info from Redis
            data = await self.redis.get(key)
            if not data:
                # No rate limit info, assume available
                return True, None

            limit_info = json.loads(data)

            # Check remaining requests
            remaining_requests = limit_info.get("remaining_requests", 0)
            remaining_tokens = limit_info.get("remaining_tokens", 0)
            reset_requests = limit_info.get("reset_requests")
            reset_tokens = limit_info.get("reset_tokens")

            now = datetime.utcnow()

            # Check if we have capacity
            has_capacity = (
                remaining_requests > 0 and remaining_tokens > 100
            )  # Keep buffer

            if has_capacity:
                return True, None

            # Calculate wait time based on earliest reset
            wait_times = []

            # Handle reset times - they could be either ISO datetime strings or duration strings
            if reset_requests:
                try:
                    # First try to parse as ISO datetime
                    reset_time = datetime.fromisoformat(reset_requests)
                    if reset_time > now:
                        wait_times.append((reset_time - now).total_seconds())
                except (ValueError, TypeError):
                    # If that fails, try to parse as duration string
                    duration = self._parse_duration_string(reset_requests)
                    wait_times.append(duration.total_seconds())

            if reset_tokens:
                try:
                    # First try to parse as ISO datetime
                    reset_time = datetime.fromisoformat(reset_tokens)
                    if reset_time > now:
                        wait_times.append((reset_time - now).total_seconds())
                except (ValueError, TypeError):
                    # If that fails, try to parse as duration string
                    duration = self._parse_duration_string(reset_tokens)
                    wait_times.append(duration.total_seconds())

            wait_seconds = int(min(wait_times)) if wait_times else 60  # Default 60s

            logger.debug(
                "rate_limit_check",
                endpoint_id=endpoint_id,
                available=False,
                wait_seconds=wait_seconds,
            )

            return False, wait_seconds

        except Exception as e:
            logger.error(
                "rate_limit_check_error", error=str(e), endpoint_id=endpoint_id
            )
            # On error, assume available to not block requests
            return True, None

    async def update_from_headers(self, endpoint_id: str, headers: dict):
        """Update rate limit info from response headers"""
        try:
            # Parse OpenAI rate limit headers
            remaining_requests = headers.get("x-ratelimit-remaining-requests")
            remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
            limit_requests = headers.get("x-ratelimit-limit-requests")
            limit_tokens = headers.get("x-ratelimit-limit-tokens")
            reset_requests = headers.get("x-ratelimit-reset-requests")
            reset_tokens = headers.get("x-ratelimit-reset-tokens")

            # Build rate limit data
            data = {"updated_at": datetime.utcnow().isoformat()}

            if remaining_requests is not None:
                data["remaining_requests"] = int(remaining_requests)
            if remaining_tokens is not None:
                data["remaining_tokens"] = int(remaining_tokens)
            if limit_requests is not None:
                data["limit_requests"] = int(limit_requests)
            if limit_tokens is not None:
                data["limit_tokens"] = int(limit_tokens)
            
            # Store reset times as duration strings - they'll be parsed when needed
            if reset_requests:
                data["reset_requests"] = reset_requests
            if reset_tokens:
                data["reset_tokens"] = reset_tokens

            # Only update if we have meaningful data
            if len(data) > 1:
                key = f"ratelimit:{endpoint_id}"
                await self.redis.setex(key, 3600, json.dumps(data))  # 1 hour TTL

                logger.debug(
                    "rate_limit_updated",
                    endpoint_id=endpoint_id,
                    remaining_requests=data.get("remaining_requests"),
                    remaining_tokens=data.get("remaining_tokens"),
                )

        except Exception as e:
            logger.error(
                "rate_limit_update_error", error=str(e), endpoint_id=endpoint_id
            )

    async def consume_capacity(self, endpoint_id: str, estimated_tokens: int = 1000):
        """
        Pre-emptively consume capacity for a request
        This helps prevent multiple concurrent requests from exceeding limits
        """
        key = f"ratelimit:{endpoint_id}"

        try:
            async with self._lock:
                data = await self.redis.get(key)
                if not data:
                    return

                limit_info = json.loads(data)

                # Decrement counters
                if "remaining_requests" in limit_info:
                    limit_info["remaining_requests"] = max(
                        0, limit_info["remaining_requests"] - 1
                    )

                if "remaining_tokens" in limit_info:
                    limit_info["remaining_tokens"] = max(
                        0, limit_info["remaining_tokens"] - estimated_tokens
                    )

                # Update Redis
                await self.redis.setex(key, 3600, json.dumps(limit_info))

        except Exception as e:
            logger.error(
                "rate_limit_consume_error", error=str(e), endpoint_id=endpoint_id
            )
