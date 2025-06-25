import hashlib
import json
from enum import Enum
from typing import Any, Dict


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
        self.is_azure = (
            "azure" in self.base_url.lower()
            if isinstance(self.base_url, str)
            else False
        )

    def _generate_deterministic_id(self, model: str, params: dict) -> str:
        """Generate a deterministic ID based on model and key parameters"""
        # Include key identifying params including API key
        id_data = {
            "model": model,
            "base_url": params.get("base_url", "openai"),
            "deployment": params.get("deployment"),  # Azure deployment name
            "api_version": params.get("api_version"),  # Azure API version
            "api_key": params.get("api_key"),
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
