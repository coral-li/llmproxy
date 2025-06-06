from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid


class EndpointStatus(Enum):
    """Endpoint health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COOLING_DOWN = "cooling_down"


class Endpoint:
    """Represents an LLM endpoint with health tracking"""

    def __init__(self, model: str, weight: int, params: dict, allowed_fails: int = 1):
        self.id = str(uuid.uuid4())
        self.model = model
        self.weight = weight
        self.params = params
        self.allowed_fails = allowed_fails

        # Health tracking
        self.status = EndpointStatus.HEALTHY
        self.consecutive_failures = 0
        self.cooldown_until: Optional[datetime] = None

        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None

        # Extract key info for logging
        self.base_url = params.get("base_url", "openai")
        self.is_azure = (
            "azure" in self.base_url.lower()
            if isinstance(self.base_url, str)
            else False
        )

    def record_success(self):
        """Record successful request"""
        self.total_requests += 1
        self.consecutive_failures = 0
        self.status = EndpointStatus.HEALTHY
        self.last_success_time = datetime.utcnow()

    def record_failure(self, error: str, cooldown_time: int = 60):
        """Record failed request and update status"""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()

        # Check if we should enter cooldown
        if self.consecutive_failures >= self.allowed_fails:
            self.status = EndpointStatus.COOLING_DOWN
            self.cooldown_until = datetime.utcnow() + timedelta(seconds=cooldown_time)

    def is_available(self) -> bool:
        """Check if endpoint is available for requests"""
        if self.status == EndpointStatus.HEALTHY:
            return True

        if self.status == EndpointStatus.COOLING_DOWN:
            if datetime.utcnow() > self.cooldown_until:
                # Exit cooldown
                self.status = EndpointStatus.HEALTHY
                self.consecutive_failures = 0
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get endpoint statistics"""
        success_rate = 0
        if self.total_requests > 0:
            success_rate = (
                (self.total_requests - self.failed_requests) / self.total_requests * 100
            )

        return {
            "id": self.id,
            "model": self.model,
            "weight": self.weight,
            "status": self.status.value,
            "base_url": self.base_url,
            "is_azure": self.is_azure,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat()
            if self.last_error_time
            else None,
            "last_success_time": self.last_success_time.isoformat()
            if self.last_success_time
            else None,
            "cooldown_until": self.cooldown_until.isoformat()
            if self.cooldown_until
            else None,
        }

    def __repr__(self):
        return f"<Endpoint {self.model} {self.base_url} status={self.status.value}>"
