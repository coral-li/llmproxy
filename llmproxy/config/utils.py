"""Configuration utilities for LLMProxy"""

from typing import Optional

# Imports are handled properly through package structure
from llmproxy.config.config_loader import load_config


def get_proxy_url(config_path: Optional[str] = None) -> str:
    """
    Get the proxy URL from the configuration file.

    Args:
        config_path: Path to the configuration file. If not provided,
                    uses LLMPROXY_CONFIG environment variable or defaults to 'llmproxy.yaml'

    Returns:
        The proxy URL in the format http://{bind_address}:{bind_port}
    """
    config = load_config(config_path)
    host = config.general_settings.bind_address
    port = config.general_settings.bind_port
    return f"http://{host}:{port}"
