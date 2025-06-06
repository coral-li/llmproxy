"""Configuration utilities for LLMProxy"""

import os
from pathlib import Path
import sys

# Add parent directory to path to import config_loader
sys.path.append(str(Path(__file__).parent.parent.parent))
from llmproxy.config.config_loader import load_config


def get_proxy_url(config_path: str = None) -> str:
    """
    Get the proxy URL from the configuration file.
    
    Args:
        config_path: Path to the configuration file. If not provided,
                    uses LLMPROXY_CONFIG environment variable or defaults to 'llmproxy.yaml'
    
    Returns:
        The proxy URL in the format http://{bind_address}:{bind_port}
    """
    if config_path is None:
        config_path = os.getenv("LLMPROXY_CONFIG", "llmproxy.yaml")
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    host = config.general_settings.bind_address
    port = config.general_settings.bind_port
    return f"http://{host}:{port}"



def get_proxy_base_url(config_path: str = None) -> str:
    """
    Get the proxy base URL for OpenAI client from the configuration file.
    
    Args:
        config_path: Path to the configuration file. If not provided,
                    uses LLMPROXY_CONFIG environment variable or defaults to 'llmproxy.yaml'
    
    Returns:
        The proxy base URL in the format http://{bind_address}:{bind_port}/v1
    """
    return f"{get_proxy_url(config_path)}/v1" 