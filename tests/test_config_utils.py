"""Tests for the config utils module"""

from unittest.mock import Mock, patch

import pytest

from llmproxy.config.utils import get_proxy_url


class TestConfigUtils:
    """Test cases for config utility functions"""

    def test_get_proxy_url_with_default_config(self):
        """Test get_proxy_url with default config path"""
        with patch("llmproxy.config.utils.load_config") as mock_load_config:
            # Mock config object
            mock_config = Mock()
            mock_config.general_settings.bind_address = "127.0.0.1"
            mock_config.general_settings.bind_port = 8000
            mock_load_config.return_value = mock_config

            result = get_proxy_url()

            assert result == "http://127.0.0.1:8000"
            mock_load_config.assert_called_once_with(None)

    def test_get_proxy_url_with_custom_config_path(self):
        """Test get_proxy_url with custom config path"""
        with patch("llmproxy.config.utils.load_config") as mock_load_config:
            # Mock config object
            mock_config = Mock()
            mock_config.general_settings.bind_address = "0.0.0.0"
            mock_config.general_settings.bind_port = 9000
            mock_load_config.return_value = mock_config

            result = get_proxy_url("custom.yaml")

            assert result == "http://0.0.0.0:9000"
            mock_load_config.assert_called_once_with("custom.yaml")

    def test_get_proxy_url_with_localhost_and_different_port(self):
        """Test get_proxy_url with localhost and different port"""
        with patch("llmproxy.config.utils.load_config") as mock_load_config:
            # Mock config object
            mock_config = Mock()
            mock_config.general_settings.bind_address = "localhost"
            mock_config.general_settings.bind_port = 3000
            mock_load_config.return_value = mock_config

            result = get_proxy_url()

            assert result == "http://localhost:3000"

    def test_get_proxy_url_with_ipv6_address(self):
        """Test get_proxy_url with IPv6 address"""
        with patch("llmproxy.config.utils.load_config") as mock_load_config:
            # Mock config object
            mock_config = Mock()
            mock_config.general_settings.bind_address = "::1"
            mock_config.general_settings.bind_port = 8080
            mock_load_config.return_value = mock_config

            result = get_proxy_url()

            assert result == "http://::1:8080"

    def test_get_proxy_url_propagates_config_errors(self):
        """Test that get_proxy_url propagates config loading errors"""
        with patch(
            "llmproxy.config.utils.load_config",
            side_effect=FileNotFoundError("Config not found"),
        ):
            with pytest.raises(FileNotFoundError):
                get_proxy_url()
