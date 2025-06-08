#!/usr/bin/env python
"""Test script to verify LLMProxy installation"""

import sys
import os
import tempfile
import subprocess

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        from llmproxy import __version__
        print(f"✓ LLMProxy version: {__version__}")
        
        from llmproxy.cli import main
        print("✓ CLI module imported successfully")
        
        from llmproxy.config.config_loader import load_config
        print("✓ Config loader imported successfully")
        
        from llmproxy.main import app
        print("✓ FastAPI app imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_loading():
    """Test config loading from current directory"""
    print("\nTesting config loading...")
    
    # Create a temporary config file
    config_content = """
model_groups:
  - model_group: test-model
    models:
      - model: test-model
        weight: 1
        params:
          api_key: test-key
          base_url: https://api.openai.com

general_settings:
  bind_address: localhost
  bind_port: 4000
  num_retries: 3
  allowed_fails: 1
  cooldown_time: 60
  redis_host: localhost
  redis_port: 6379
  redis_password: ""
  cache: false
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config = f.name
    
    try:
        from llmproxy.config.config_loader import load_config
        config = load_config(temp_config)
        print(f"✓ Config loaded successfully")
        print(f"  - Model groups: {len(config.model_groups)}")
        print(f"  - Bind address: {config.general_settings.bind_address}")
        print(f"  - Bind port: {config.general_settings.bind_port}")
        return True
    except Exception as e:
        print(f"✗ Config loading error: {e}")
        return False
    finally:
        os.unlink(temp_config)

def test_cli_help():
    """Test that the CLI help works"""
    print("\nTesting CLI help...")
    
    try:
        from llmproxy.cli import main
        # This would normally call sys.exit, so we can't test it directly
        # But we can at least verify the function exists
        print("✓ CLI main function accessible")
        return True
    except Exception as e:
        print(f"✗ CLI error: {e}")
        return False

def main():
    """Run all tests"""
    print("LLMProxy Installation Test")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_config_loading,
        test_cli_help,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! LLMProxy is ready to use.")
        print("\nNext steps:")
        print("1. Copy llmproxy.yaml.example to llmproxy.yaml")
        print("2. Configure your API keys and endpoints")
        print("3. Start Redis server")
        print("4. Run: llmproxy")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 