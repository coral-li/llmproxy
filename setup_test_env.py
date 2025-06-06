#!/usr/bin/env python
"""Setup test environment for LLMProxy"""

import os
from pathlib import Path

def create_test_env():
    """Create a test .env file with dummy values"""
    
    test_env_content = """# Test environment for LLMProxy
# This uses the llmproxy.test.yaml configuration

# Redis settings (assumes local Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Use test configuration
LLMPROXY_CONFIG=llmproxy.test.yaml

# Optional settings
LOG_LEVEL=INFO
"""
    
    env_path = Path('.env')
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Overwrite with test configuration? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    with open(env_path, 'w') as f:
        f.write(test_env_content)
    
    print("✅ Created test .env file")
    print("📝 Using llmproxy.test.yaml configuration")
    print("\nTo run with real endpoints:")
    print("1. Copy env.example to .env")
    print("2. Fill in your actual API keys")
    print("3. Use the default llmproxy.yaml configuration")

if __name__ == "__main__":
    create_test_env() 