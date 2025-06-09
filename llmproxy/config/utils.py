"""Configuration utilities for LLMProxy"""

import asyncio
from typing import Optional

# Imports are handled properly through package structure
from llmproxy.config.config_loader import load_config_async


async def get_proxy_url_async(config_path: Optional[str] = None) -> str:
    """
    Get the proxy URL from the configuration file.

    Args:
        config_path: Path to the configuration file. If not provided,
                    uses LLMPROXY_CONFIG environment variable or defaults to 'llmproxy.yaml'

    Returns:
        The proxy URL in the format http://{bind_address}:{bind_port}
    """
    config = await load_config_async(config_path)
    host = config.general_settings.bind_address
    port = config.general_settings.bind_port
    return f"http://{host}:{port}"


def get_proxy_url(config_path: Optional[str] = None) -> str:
    """
    Synchronous wrapper for get_proxy_url_async for backward compatibility.

    Args:
        config_path: Path to the configuration file. If not provided,
                    uses LLMPROXY_CONFIG environment variable or defaults to 'llmproxy.yaml'

    Returns:
        The proxy URL in the format http://{bind_address}:{bind_port}
    """
    return asyncio.run(get_proxy_url_async(config_path))
