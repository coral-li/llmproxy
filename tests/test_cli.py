"""Tests for the CLI module"""

from unittest.mock import Mock, patch

import pytest

from llmproxy.cli import main


class TestCLI:
    """Test cases for CLI functionality"""

    def test_main_with_default_args(self):
        """Test main function with default arguments"""
        with (
            patch("sys.argv", ["llmproxy"]),
            patch(
                "llmproxy.cli.load_config_async", new=Mock()
            ) as mock_load_config_async,
            patch("uvicorn.Server.run") as mock_run,
            patch("llmproxy.cli.setup_logging") as mock_setup_logging,
            patch("os.path.exists", return_value=False),
            patch("asyncio.run") as mock_asyncio_run,
        ):
            # Mock config object
            mock_config = Mock()
            mock_config.general_settings.bind_address = "127.0.0.1"
            mock_config.general_settings.bind_port = 8000
            mock_load_config_async.return_value = mock_config
            mock_asyncio_run.return_value = mock_config

            main()

            mock_setup_logging.assert_called_once_with("INFO")
            mock_asyncio_run.assert_called_once()
            mock_run.assert_called_once()

    def test_main_with_custom_config(self):
        """Test main function with custom config file"""
        with (
            patch("sys.argv", ["llmproxy", "--config", "custom.yaml"]),
            patch(
                "llmproxy.cli.load_config_async", new=Mock()
            ) as mock_load_config_async,
            patch("uvicorn.Server.run") as mock_run,
            patch("llmproxy.cli.setup_logging"),
            patch("os.path.exists", return_value=False),
            patch("asyncio.run") as mock_asyncio_run,
        ):
            mock_config = Mock()
            mock_config.general_settings.bind_address = "0.0.0.0"
            mock_config.general_settings.bind_port = 9000
            mock_load_config_async.return_value = mock_config
            mock_asyncio_run.return_value = mock_config

            main()

            mock_asyncio_run.assert_called_once()
            mock_run.assert_called_once()

    def test_main_with_host_port_override(self):
        """Test main function with host and port overrides"""
        with (
            patch("sys.argv", ["llmproxy", "--host", "192.168.1.1", "--port", "3000"]),
            patch(
                "llmproxy.cli.load_config_async", new=Mock()
            ) as mock_load_config_async,
            patch("uvicorn.Server.run") as mock_run,
            patch("llmproxy.cli.setup_logging"),
            patch("os.path.exists", return_value=False),
            patch("asyncio.run") as mock_asyncio_run,
        ):
            mock_config = Mock()
            mock_config.general_settings.bind_address = "127.0.0.1"
            mock_config.general_settings.bind_port = 8000
            mock_load_config_async.return_value = mock_config
            mock_asyncio_run.return_value = mock_config

            main()

            # Verify the config was loaded correctly
            mock_asyncio_run.assert_called_once()
            mock_run.assert_called_once()

    def test_main_with_log_level(self):
        """Test main function with different log levels"""
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for log_level in log_levels:
            with (
                patch("sys.argv", ["llmproxy", "--log-level", log_level]),
                patch(
                    "llmproxy.cli.load_config_async", new=Mock()
                ) as mock_load_config_async,
                patch("uvicorn.Server.run"),
                patch("llmproxy.cli.setup_logging") as mock_setup_logging,
                patch("os.path.exists", return_value=False),
                patch("asyncio.run") as mock_asyncio_run,
            ):
                mock_config = Mock()
                mock_config.general_settings.bind_address = "127.0.0.1"
                mock_config.general_settings.bind_port = 8000
                mock_load_config_async.return_value = mock_config
                mock_asyncio_run.return_value = mock_config

                main()

                mock_setup_logging.assert_called_with(log_level)

    def test_main_loads_dotenv_when_exists(self):
        """Test main function loads .env file when it exists"""
        with (
            patch("sys.argv", ["llmproxy"]),
            patch(
                "llmproxy.cli.load_config_async", new=Mock()
            ) as mock_load_config_async,
            patch("uvicorn.Server.run"),
            patch("llmproxy.cli.setup_logging"),
            patch("os.path.exists", return_value=True),
            patch("llmproxy.cli.load_dotenv") as mock_load_dotenv,
            patch("asyncio.run") as mock_asyncio_run,
        ):
            mock_config = Mock()
            mock_config.general_settings.bind_address = "127.0.0.1"
            mock_config.general_settings.bind_port = 8000
            mock_load_config_async.return_value = mock_config
            mock_asyncio_run.return_value = mock_config

            main()

            mock_load_dotenv.assert_called_once()

    def test_main_handles_keyboard_interrupt(self):
        """Test main function handles KeyboardInterrupt gracefully"""
        with (
            patch("sys.argv", ["llmproxy"]),
            patch(
                "llmproxy.cli.load_config_async", new=Mock()
            ) as mock_load_config_async,
            patch("uvicorn.Server.run", side_effect=KeyboardInterrupt),
            patch("llmproxy.cli.setup_logging"),
            patch("os.path.exists", return_value=False),
            patch("sys.exit") as mock_exit,
            patch("asyncio.run") as mock_asyncio_run,
        ):
            mock_config = Mock()
            mock_config.general_settings.bind_address = "127.0.0.1"
            mock_config.general_settings.bind_port = 8000
            mock_load_config_async.return_value = mock_config
            mock_asyncio_run.return_value = mock_config

            main()

            mock_exit.assert_called_once_with(0)

    def test_main_handles_file_not_found_error(self):
        """Test main function handles FileNotFoundError"""
        with (
            patch("sys.argv", ["llmproxy"]),
            patch("llmproxy.cli.load_config_async", new=Mock()),
            patch(
                "asyncio.run",
                side_effect=FileNotFoundError("Config not found"),
            ),
            patch("llmproxy.cli.setup_logging"),
            patch("os.path.exists", return_value=False),
            patch("sys.exit") as mock_exit,
        ):
            main()

            mock_exit.assert_called_once_with(1)

    def test_main_handles_general_exception(self):
        """Test main function handles general exceptions"""
        with (
            patch("sys.argv", ["llmproxy"]),
            patch("llmproxy.cli.load_config_async", new=Mock()),
            patch("asyncio.run", side_effect=Exception("General error")),
            patch("llmproxy.cli.setup_logging"),
            patch("os.path.exists", return_value=False),
            patch("sys.exit") as mock_exit,
        ):
            main()

            mock_exit.assert_called_once_with(1)

    def test_argument_parser_setup(self):
        """Test that argument parser is set up correctly"""
        with patch("sys.argv", ["llmproxy", "--help"]), pytest.raises(SystemExit):
            main()

    def test_main_with_all_arguments(self):
        """Test main function with all possible arguments"""
        with (
            patch(
                "sys.argv",
                [
                    "llmproxy",
                    "--config",
                    "test.yaml",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "5000",
                    "--log-level",
                    "DEBUG",
                ],
            ),
            patch(
                "llmproxy.cli.load_config_async", new=Mock()
            ) as mock_load_config_async,
            patch("uvicorn.Server.run"),
            patch("llmproxy.cli.setup_logging") as mock_setup_logging,
            patch("os.path.exists", return_value=True),
            patch("llmproxy.cli.load_dotenv"),
            patch("asyncio.run") as mock_asyncio_run,
        ):
            mock_config = Mock()
            mock_config.general_settings.bind_address = "127.0.0.1"
            mock_config.general_settings.bind_port = 8000
            mock_load_config_async.return_value = mock_config
            mock_asyncio_run.return_value = mock_config

            main()

            mock_setup_logging.assert_called_once_with("DEBUG")
            mock_asyncio_run.assert_called_once()
