import asyncio
import random
from typing import Optional, Callable, Any, TypeVar
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryHandler:
    """Handles retry logic with exponential backoff"""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute_with_retry(
        self, func: Callable[..., T], *args, retry_on: Optional[tuple] = None, **kwargs
    ) -> T:
        """
        Execute function with exponential backoff retry

        Args:
            func: Async function to execute
            retry_on: Tuple of exception types to retry on. If None, retry on all exceptions
            *args, **kwargs: Arguments to pass to func
        """
        last_error = None
        retry_on = retry_on or (Exception,)

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)

            except retry_on as e:
                last_error = e

                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter

                    logger.warning(
                        "retry_attempt",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        delay=total_delay,
                        error=str(e),
                    )

                    await asyncio.sleep(total_delay)
                else:
                    # Last attempt failed
                    logger.error(
                        "retry_exhausted", attempts=self.max_retries, error=str(e)
                    )

        # Re-raise the last error
        raise last_error


class APIError(Exception):
    """Base exception for API errors"""

    def __init__(
        self, message: str, status_code: int = 500, error_type: str = "api_error"
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert error to OpenAI-compatible error format"""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.status_code,
            }
        }


class RateLimitError(APIError):
    """Rate limit exceeded error"""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None
    ):
        super().__init__(message, 429, "rate_limit_error")
        self.retry_after = retry_after


class EndpointError(APIError):
    """Endpoint-specific error"""

    def __init__(self, message: str, endpoint_id: str, status_code: int = 500):
        super().__init__(message, status_code, "endpoint_error")
        self.endpoint_id = endpoint_id


def is_retryable_error(status_code: int) -> bool:
    """Determine if an HTTP status code indicates a retryable error"""
    # Retry on:
    # - 429: Rate limit (though we try to avoid these)
    # - 500-599: Server errors
    # - 408: Request timeout
    # - 502: Bad gateway
    # - 503: Service unavailable
    # - 504: Gateway timeout
    return status_code in {408, 429} or 500 <= status_code < 600
