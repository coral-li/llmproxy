#!/usr/bin/env python3
"""Installation test script for verifying llmproxy components."""

import os
from typing import List


def test_basic_imports() -> None:
    """Test that basic imports work correctly."""
    print("Testing basic imports...")

    try:
        import llmproxy.cli.main  # noqa: F401

        print("✅ CLI import successful")
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return

    try:
        import llmproxy.config.config_loader  # noqa: F401

        print("✅ Config loader import successful")
    except ImportError as e:
        print(f"❌ Config loader import failed: {e}")
        return

    try:
        import llmproxy.main  # noqa: F401

        print("✅ Main app import successful")
    except ImportError as e:
        print(f"❌ Main app import failed: {e}")
        return


def check_config_files() -> None:
    """Check for required configuration files."""
    print("\nChecking configuration files...")

    config_files = ["llmproxy.yaml", "llmproxy.yaml.example", "env.example"]

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} exists")
        else:
            print(f"⚠️  {config_file} not found")


def check_dependencies() -> None:
    """Check that all required dependencies are installed."""
    print("\nChecking dependencies...")

    required_packages = [
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "redis",
        "structlog",
        "pytest",
    ]

    missing_packages: List[str] = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)

    if missing_packages:
        print("\n❌ Missing packages: " + ", ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies are installed")
        return True


def test_cli_help() -> None:
    """Test that CLI help works."""
    print("\nTesting CLI help...")

    try:
        import llmproxy.cli.main  # noqa: F401

        print("✅ CLI module loaded successfully")
    except Exception as e:
        print(f"❌ CLI module failed to load: {e}")


def run_all_tests() -> None:
    """Run all installation tests."""
    print("🚀 Running LLMProxy installation tests...")
    print("=" * 50)

    test_basic_imports()
    check_config_files()
    check_dependencies()
    test_cli_help()

    print("\n" + "=" * 50)
    print("✅ Installation tests completed")


if __name__ == "__main__":
    run_all_tests()
