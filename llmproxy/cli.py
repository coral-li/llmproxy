#!/usr/bin/env python
"""Command line interface for LLMProxy"""

import argparse
import os
import sys

import uvicorn
from dotenv import load_dotenv

from llmproxy.config.config_loader import load_config
from llmproxy.core.logger import setup_logging


def main() -> None:
    """Main entry point for the llmproxy command"""
    parser = argparse.ArgumentParser(
        description="LLMProxy - Intelligent load balancer for LLM APIs"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (default: llmproxy.yaml in current directory)",
    )
    parser.add_argument("--host", type=str, help="Host to bind to (overrides config)")
    parser.add_argument("--port", type=int, help="Port to bind to (overrides config)")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging with the specified level
    setup_logging(args.log_level)

    # Load environment variables from .env file if it exists
    if os.path.exists(".env"):
        load_dotenv()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override with command line arguments if provided
        host = args.host or config.general_settings.bind_address
        port = args.port or config.general_settings.bind_port

        print(f"Starting LLMProxy on {host}:{port}")
        print(f"Configuration loaded from: {args.config or 'llmproxy.yaml'}")

        # Create uvicorn config with proper signal handling
        config_obj = uvicorn.Config(
            "llmproxy.main:app",
            host=host,
            port=port,
            log_level=args.log_level.lower(),
            access_log=False,
            server_header=False,
            use_colors=True,
            reload=False,
        )

        # Create server and run with proper signal handling
        server = uvicorn.Server(config_obj)
        server.run()

    except KeyboardInterrupt:
        # Handle CTRL+C gracefully without ugly traceback
        print("\nLLMProxy shutdown complete.")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure llmproxy.yaml exists in the current directory or specify --config"
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error starting LLMProxy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
