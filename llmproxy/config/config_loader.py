import yaml
import os
from typing import Any, Dict, List, Union
from pathlib import Path
import sys

# Add parent directory to path to import config_model
sys.path.append(str(Path(__file__).parent.parent.parent))
from config_model import LLMProxyConfig


def resolve_env_vars(data: Any) -> Any:
    """
    Recursively resolve environment variable references in the format 'os.environ/VAR_NAME'
    Raises ValueError if any environment variable is not set.
    """
    missing_vars = []

    def _resolve(data: Any) -> Any:
        if isinstance(data, dict):
            return {key: _resolve(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [_resolve(item) for item in data]
        elif isinstance(data, str) and data.startswith("os.environ/"):
            env_var = data.replace("os.environ/", "")
            env_value = os.getenv(env_var)
            if env_value is None:
                missing_vars.append(env_var)
                return data  # Return original for now
            return env_value
        else:
            return data

    resolved_data = _resolve(data)

    if missing_vars:
        unique_missing = sorted(set(missing_vars))
        raise ValueError(
            f"The following environment variables are not set: {', '.join(unique_missing)}"
        )

    return resolved_data


def load_config() -> LLMProxyConfig:
    """
    Load and validate YAML configuration using the Pydantic model

    Args:
        yaml_file_path: Path to the YAML configuration file

    Returns:
        Validated LLMProxyConfig instance
    """
    yaml_file_path = "llmproxy.yaml"
    # Check if file exists
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_file_path}")

    with open(yaml_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Replace environment variable references with actual values
    yaml_data = resolve_env_vars(yaml_data)

    # If cache_params doesn't have host/port/password, inherit from general_settings
    if yaml_data.get("general_settings", {}).get("cache_params"):
        cache_params = yaml_data["general_settings"]["cache_params"]
        general = yaml_data["general_settings"]

        # Set defaults from general settings if not specified
        if "host" not in cache_params:
            cache_params["host"] = general.get("redis_host")
        if "port" not in cache_params:
            cache_params["port"] = general.get("redis_port")
        if "password" not in cache_params:
            cache_params["password"] = general.get("redis_password")

    # Create and validate the configuration
    config = LLMProxyConfig(**yaml_data)
    return config
