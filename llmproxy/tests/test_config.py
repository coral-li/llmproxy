import pytest
import os
import tempfile
import yaml
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llmproxy.config.config_loader import load_config, resolve_env_vars


class TestConfigLoader:
    """Tests for configuration loading"""

    def test_resolve_env_vars(self):
        """Test environment variable resolution"""
        # Set test env vars
        os.environ["TEST_VAR"] = "test_value"
        os.environ["TEST_PORT"] = "8080"

        data = {
            "key1": "os.environ/TEST_VAR",
            "key2": "regular_value",
            "nested": {"port": "os.environ/TEST_PORT"},
            "list": ["item1", "os.environ/TEST_VAR"],
        }

        resolved = resolve_env_vars(data)

        assert resolved["key1"] == "test_value"
        assert resolved["key2"] == "regular_value"
        assert resolved["nested"]["port"] == "8080"
        assert resolved["list"][1] == "test_value"

        # Cleanup
        del os.environ["TEST_VAR"]
        del os.environ["TEST_PORT"]

    def test_missing_env_var(self):
        """Test error handling for missing environment variables"""
        data = {"key": "os.environ/MISSING_VAR"}

        with pytest.raises(ValueError) as exc_info:
            resolve_env_vars(data)

        assert "MISSING_VAR" in str(exc_info.value)

    def test_load_config(self):
        """Test loading configuration from YAML file"""
        # Create temporary config file
        config_data = {
            "model_groups": [
                {
                    "model_group": "test-model",
                    "models": [
                        {
                            "model": "test-model",
                            "weight": 1,
                            "params": {
                                "api_key": "test-key",
                                "base_url": "http://test.com",
                            },
                        }
                    ],
                }
            ],
            "general_settings": {
                "bind_address": "127.0.0.1",
                "bind_port": 5000,
                "num_retries": 3,
                "allowed_fails": 1,
                "cooldown_time": 60,
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_password": "",
                "cache": True,
                "cache_params": {"type": "redis", "ttl": 300, "namespace": "test"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Load config
            config = load_config(temp_path)

            # Verify loaded config
            assert len(config.model_groups) == 1
            assert config.model_groups[0].model_group == "test-model"
            assert config.general_settings.bind_port == 5000
            assert config.general_settings.cache is True
            assert config.general_settings.cache_params.ttl == 300

        finally:
            # Cleanup
            os.unlink(temp_path)

    def test_cache_params_inheritance(self):
        """Test that cache_params inherits from general_settings"""
        config_data = {
            "model_groups": [
                {
                    "model_group": "test",
                    "models": [
                        {"model": "test", "weight": 1, "params": {"api_key": "key"}}
                    ],
                }
            ],
            "general_settings": {
                "redis_host": "redis.example.com",
                "redis_port": 6380,
                "redis_password": "secret",
                "cache": True,
                "cache_params": {
                    "type": "redis",
                    "ttl": 3600,
                    "namespace": "prod"
                    # Note: not specifying host/port/password
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            # Verify cache params inherited from general settings
            assert config.general_settings.cache_params.host == "redis.example.com"
            assert config.general_settings.cache_params.port == 6380
            assert config.general_settings.cache_params.password == "secret"

        finally:
            os.unlink(temp_path)
