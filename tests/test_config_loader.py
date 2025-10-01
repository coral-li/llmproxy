"""Tests for the config loader module"""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from llmproxy.config.config_loader import load_config_async, resolve_env_vars
from llmproxy.config_model import LLMProxyConfig


class TestResolveEnvVars:
    """Test cases for resolve_env_vars function"""

    def test_resolve_env_vars_simple_string(self):
        """Test resolving environment variables in simple strings"""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            data = "os.environ/TEST_VAR"
            result = resolve_env_vars(data)
            assert result == "test_value"

    def test_resolve_env_vars_in_dict(self):
        """Test resolving environment variables in dictionaries"""
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
            data = {
                "server": {"host": "os.environ/HOST", "port": "os.environ/PORT"},
                "other": "static_value",
            }
            result = resolve_env_vars(data)
            expected = {
                "server": {"host": "localhost", "port": "8080"},
                "other": "static_value",
            }
            assert result == expected

    def test_resolve_env_vars_in_list(self):
        """Test resolving environment variables in lists"""
        with patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"}):
            data = ["os.environ/VAR1", "static", "os.environ/VAR2"]
            result = resolve_env_vars(data)
            assert result == ["value1", "static", "value2"]

    def test_resolve_env_vars_nested_structures(self):
        """Test resolving environment variables in nested structures"""
        with patch.dict(os.environ, {"API_KEY": "secret123", "DEBUG": "true"}):
            data = {
                "config": {
                    "api_key": "os.environ/API_KEY",
                    "settings": ["os.environ/DEBUG", "production"],
                }
            }
            result = resolve_env_vars(data)
            expected = {
                "config": {"api_key": "secret123", "settings": ["true", "production"]}
            }
            assert result == expected

    def test_resolve_env_vars_missing_variable(self):
        """Test error when environment variable is missing"""
        data = {"value": "os.environ/MISSING_VAR"}
        with pytest.raises(
            ValueError,
            match="The following environment variables are not set: MISSING_VAR",
        ):
            resolve_env_vars(data)

    def test_resolve_env_vars_multiple_missing_variables(self):
        """Test error with multiple missing environment variables"""
        data = {
            "var1": "os.environ/MISSING_VAR1",
            "var2": "os.environ/MISSING_VAR2",
            "var3": "os.environ/MISSING_VAR1",  # Duplicate should be listed once
        }
        with pytest.raises(
            ValueError,
            match="The following environment variables are not set: MISSING_VAR1, MISSING_VAR2",
        ):
            resolve_env_vars(data)

    def test_resolve_env_vars_non_env_strings(self):
        """Test that non-environment variable strings are left unchanged"""
        data = {
            "normal_string": "hello world",
            "prefix_string": "prefix_os.environ/VAR",
            "number": 42,
            "boolean": True,
        }
        result = resolve_env_vars(data)
        assert result == data

    def test_resolve_env_vars_empty_data(self):
        """Test with empty or None data"""
        assert resolve_env_vars({}) == {}
        assert resolve_env_vars([]) == []
        assert resolve_env_vars(None) is None
        assert resolve_env_vars("") == ""


class TestLoadConfig:
    """Test cases for load_config_async function"""

    def create_test_config_file(self, content: dict) -> str:
        """Helper to create a temporary config file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(content, temp_file)
        temp_file.flush()
        return temp_file.name

    async def test_load_config_with_explicit_path(self):
        """Test loading config with explicit path"""
        config_content = {
            "general_settings": {
                "bind_address": "127.0.0.1",
                "bind_port": 8000,
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_password": "secret",
            },
            "model_groups": [
                {
                    "model_group": "gpt-3.5-turbo",
                    "models": [
                        {
                            "model": "gpt-3.5-turbo",
                            "weight": 1,
                            "params": {
                                "api_key": "test-key",
                                "base_url": "https://api.openai.com",
                            },
                        }
                    ],
                }
            ],
        }

        config_path = self.create_test_config_file(config_content)
        try:
            config = await load_config_async(config_path)
            assert isinstance(config, LLMProxyConfig)
            assert config.general_settings.bind_address == "127.0.0.1"
            assert config.general_settings.bind_port == 8000
            assert config.general_settings.http_timeout == 300.0
        finally:
            os.unlink(config_path)

    async def test_load_config_with_env_var_path(self):
        """Test loading config using LLMPROXY_CONFIG environment variable"""
        config_content = {
            "general_settings": {
                "bind_address": "0.0.0.0",
                "bind_port": 9000,
                "redis_host": "redis.example.com",
                "redis_port": 6379,
                "redis_password": "secret",
            },
            "model_groups": [{"model_group": "test-model", "models": []}],
        }

        config_path = self.create_test_config_file(config_content)
        try:
            with patch.dict(os.environ, {"LLMPROXY_CONFIG": config_path}):
                config = await load_config_async()
                assert config.general_settings.bind_address == "0.0.0.0"
                assert config.general_settings.bind_port == 9000
                assert config.general_settings.http_timeout == 300.0
        finally:
            os.unlink(config_path)

    async def test_load_config_default_path(self):
        """Test loading config from default path"""
        config_content = {
            "general_settings": {
                "bind_address": "localhost",
                "bind_port": 3000,
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_password": "",
            },
            "model_groups": [{"model_group": "default", "models": []}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "llmproxy.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config_content, f)

            with patch("os.getcwd", return_value=temp_dir), patch.dict(
                os.environ, {}, clear=True
            ):  # Clear LLMPROXY_CONFIG
                config = await load_config_async()
                assert config.general_settings.bind_address == "localhost"
                assert config.general_settings.bind_port == 3000
                assert config.general_settings.http_timeout == 300.0

    async def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist"""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            await load_config_async("/nonexistent/path/config.yaml")

    async def test_load_config_with_env_vars(self):
        """Test loading config with environment variable substitution"""
        config_content = {
            "general_settings": {
                "bind_address": "os.environ/BIND_HOST",
                "bind_port": "os.environ/BIND_PORT",
                "redis_host": "os.environ/REDIS_HOST",
                "redis_port": "os.environ/REDIS_PORT",
                "redis_password": "os.environ/REDIS_PASSWORD",
                "http_timeout": "os.environ/HTTP_TIMEOUT",
            },
            "model_groups": [
                {
                    "model_group": "test",
                    "models": [
                        {
                            "model": "gpt-3.5-turbo",
                            "weight": 1,
                            "params": {
                                "api_key": "os.environ/API_KEY",
                                "base_url": "os.environ/API_URL",
                            },
                        }
                    ],
                }
            ],
        }

        config_path = self.create_test_config_file(config_content)
        try:
            with patch.dict(
                os.environ,
                {
                    "BIND_HOST": "0.0.0.0",
                    "BIND_PORT": "8080",
                    "REDIS_HOST": "redis.example.com",
                    "REDIS_PORT": "6379",
                    "REDIS_PASSWORD": "redis_secret",
                    "HTTP_TIMEOUT": "150",
                    "API_URL": "https://api.example.com",
                    "API_KEY": "secret-key",
                },
            ):
                config = await load_config_async(config_path)
                assert config.general_settings.bind_address == "0.0.0.0"
                # Note: bind_port will be converted to int by pydantic
                assert config.general_settings.bind_port == 8080
                assert config.general_settings.http_timeout == 150.0
        finally:
            os.unlink(config_path)

    async def test_load_config_cache_params_inheritance(self):
        """Test cache params inheriting from general settings"""
        config_content = {
            "general_settings": {
                "bind_address": "127.0.0.1",
                "bind_port": 8000,
                "redis_host": "redis.example.com",
                "redis_port": 6379,
                "redis_password": "redis_secret",
                "cache_params": {
                    # Missing host, port, password - should inherit
                    "ttl": 3600
                },
            },
            "model_groups": [{"model_group": "test", "models": []}],
        }

        config_path = self.create_test_config_file(config_content)
        try:
            config = await load_config_async(config_path)
            cache_params = config.general_settings.cache_params
            assert cache_params.host == "redis.example.com"
            assert cache_params.port == 6379
            assert cache_params.password == "redis_secret"
            assert cache_params.ttl == 3600
        finally:
            os.unlink(config_path)

    async def test_load_config_cache_params_explicit_override(self):
        """Test cache params with explicit values (no inheritance)"""
        config_content = {
            "general_settings": {
                "bind_address": "127.0.0.1",
                "bind_port": 8000,
                "redis_host": "redis.example.com",
                "redis_port": 6379,
                "redis_password": "redis_secret",
                "cache_params": {
                    "host": "cache.example.com",
                    "port": 6380,
                    "password": "cache_secret",
                    "ttl": 1800,
                },
            },
            "model_groups": [{"model_group": "test", "models": []}],
        }

        config_path = self.create_test_config_file(config_content)
        try:
            config = await load_config_async(config_path)
            cache_params = config.general_settings.cache_params
            assert cache_params.host == "cache.example.com"
            assert cache_params.port == 6380
            assert cache_params.password == "cache_secret"
            assert cache_params.ttl == 1800
        finally:
            os.unlink(config_path)

    async def test_load_config_no_cache_params(self):
        """Test loading config without cache_params section"""
        config_content = {
            "general_settings": {
                "bind_address": "127.0.0.1",
                "bind_port": 8000,
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_password": "",
            },
            "model_groups": [{"model_group": "test", "models": []}],
        }

        config_path = self.create_test_config_file(config_content)
        try:
            config = await load_config_async(config_path)
            # Should load successfully without cache_params
            assert config.general_settings.bind_address == "127.0.0.1"
        finally:
            os.unlink(config_path)

    async def test_load_config_invalid_yaml(self):
        """Test error with invalid YAML"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        temp_file.write("invalid: yaml: content: [\n")
        temp_file.flush()

        try:
            with pytest.raises(yaml.YAMLError):
                await load_config_async(temp_file.name)
        finally:
            os.unlink(temp_file.name)

    async def test_load_config_validation_error(self):
        """Test error with invalid config structure"""
        config_content = {
            "general_settings": {"bind_port": "invalid_port"}  # Should be integer
        }

        config_path = self.create_test_config_file(config_content)
        try:
            with pytest.raises(Exception):  # Pydantic validation error
                await load_config_async(config_path)
        finally:
            os.unlink(config_path)
