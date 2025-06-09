#!/usr/bin/env python3
"""Test script for verifying proxy configuration."""

import os
import sys
from typing import Optional

from llmproxy.config.config_loader import load_config

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_proxy_url(proxy_url: Optional[str] = None) -> str:
    """Get proxy URL from environment or parameter."""
    if proxy_url is not None:
        return proxy_url

    proxy_url_env = os.getenv("PROXY_URL", "http://localhost:8000")
    return proxy_url_env


def test_proxy_config() -> None:
    """Test proxy configuration loading."""
    try:
        config = load_config("llmproxy.yaml")
        print("✅ Configuration loaded successfully")
        print(f"   Model groups: {list(config.model_groups.keys())}")
        print(f"   General settings: {config.general_settings}")

        # Test proxy URL
        proxy_url = get_proxy_url()
        print(f"   Proxy URL: {proxy_url}")

    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_proxy_config()
