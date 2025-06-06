#!/usr/bin/env python
"""Configuration helper for test scripts"""

import httpx
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
from llmproxy.config.utils import get_proxy_url


def get_configured_model(proxy_url: str = None) -> Optional[str]:
    """Get a configured model from the proxy"""
    if proxy_url is None:
        proxy_url = get_proxy_url()
    
    # Check health endpoint to get configured models
    response = httpx.get(f"{proxy_url}/health", timeout=2)
    if response.status_code == 200:
        data = response.json()
        model_groups = data.get('model_groups', [])
        if model_groups:
            # Return the first available model
            return model_groups[0]



def get_test_config():
    """Get test configuration"""
    # Allow override via environment variable, otherwise get from config
    proxy_url = os.getenv('LLMPROXY_URL')
    if proxy_url is None:
        proxy_url = get_proxy_url()
    
    model = get_configured_model(proxy_url)
    
    return {
        'proxy_url': proxy_url,
        'model': model,
        'quick_test': os.getenv('LLMPROXY_QUICK_TEST', '').lower() == 'true'
    }


if __name__ == "__main__":
    config = get_test_config()
    print(f"Test configuration:")
    print(f"  Proxy URL: {config['proxy_url']}")
    print(f"  Model: {config['model']}")
    print(f"  Quick test: {config['quick_test']}") 