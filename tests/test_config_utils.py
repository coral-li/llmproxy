"""Tests for the config utils module"""

from unittest.mock import patch

import pytest

from llmproxy.config.utils import get_proxy_url


class TestConfigUtils:
    """Test cases for config utility functions"""

    def test_get_proxy_url_with_default_config(self):
        """Test get_proxy_url with default config path"""
        with patch(
            "llmproxy.config.utils.get_proxy_url_async"
        ) as mock_get_proxy_url_async:
            # Mock the async function to return the expected URL
            mock_get_proxy_url_async.return_value = "http://127.0.0.1:8000"

            result = get_proxy_url()

            assert result == "http://127.0.0.1:8000"
            mock_get_proxy_url_async.assert_called_once_with(None)

    def test_get_proxy_url_with_custom_config_path(self):
        """Test get_proxy_url with custom config path"""
        with patch(
            "llmproxy.config.utils.get_proxy_url_async"
        ) as mock_get_proxy_url_async:
            # Mock the async function to return the expected URL
            mock_get_proxy_url_async.return_value = "http://0.0.0.0:9000"

            result = get_proxy_url("custom.yaml")

            assert result == "http://0.0.0.0:9000"
            mock_get_proxy_url_async.assert_called_once_with("custom.yaml")

    def test_get_proxy_url_with_localhost_and_different_port(self):
        """Test get_proxy_url with localhost and different port"""
        with patch(
            "llmproxy.config.utils.get_proxy_url_async"
        ) as mock_get_proxy_url_async:
            # Mock the async function to return the expected URL
            mock_get_proxy_url_async.return_value = "http://localhost:3000"

            result = get_proxy_url()

            assert result == "http://localhost:3000"

    def test_get_proxy_url_with_ipv6_address(self):
        """Test get_proxy_url with IPv6 address"""
        with patch(
            "llmproxy.config.utils.get_proxy_url_async"
        ) as mock_get_proxy_url_async:
            # Mock the async function to return the expected URL
            mock_get_proxy_url_async.return_value = "http://::1:8080"

            result = get_proxy_url()

            assert result == "http://::1:8080"

    def test_get_proxy_url_propagates_config_errors(self):
        """Test that get_proxy_url propagates config loading errors"""
        with patch(
            "llmproxy.config.utils.get_proxy_url_async",
            side_effect=FileNotFoundError("Config not found"),
        ):
            with pytest.raises(FileNotFoundError):
                get_proxy_url()
