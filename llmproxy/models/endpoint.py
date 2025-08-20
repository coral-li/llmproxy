import hashlib
import json
from enum import Enum
from typing import Any, Dict

from llmproxy.core.azure_utils import is_azure_host
from llmproxy.core.logger import get_logger

logger = get_logger(__name__)


class EndpointStatus(Enum):
    """Endpoint health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COOLING_DOWN = "cooling_down"


class Endpoint:
    """Represents an LLM endpoint configuration (stateless)"""

    def __init__(self, model: str, weight: int, params: dict, allowed_fails: int = 1):
        # Generate deterministic ID based on model and key params
        self.id = self._generate_deterministic_id(model, params)
        self.model = model
        self.weight = weight
        self.params = params
        self.allowed_fails = allowed_fails

        # Extract key info for logging (config only, no state)
        self.base_url = params.get("base_url", "openai")
        assert isinstance(
            self.base_url, str
        ), f"base_url must be a string but is a {type(self.base_url)}"
        self.is_azure = is_azure_host(self.base_url)

    def _generate_deterministic_id(self, model: str, params: dict) -> str:
        """Generate a deterministic ID based on model and key parameters"""
        # Include key identifying params including API key
        id_data = {
            "model": model,
            "base_url": params.get("base_url", "openai"),
            "default_query": params.get("default_query"),
        }

        # Remove None values
        id_data = {k: v for k, v in id_data.items() if v is not None}

        # Create deterministic hash
        id_str = json.dumps(id_data, sort_keys=True)

        return hashlib.sha256(id_str.encode()).hexdigest()[:16]

    def get_config_dict(self) -> Dict[str, Any]:
        """Get endpoint configuration (no state, just config)"""
        return {
            "id": self.id,
            "model": self.model,
            "weight": self.weight,
            "base_url": self.base_url,
            "is_azure": self.is_azure,
            "allowed_fails": self.allowed_fails,
        }

    def __repr__(self) -> str:
        return f"<Endpoint {self.model} {self.base_url}>"
