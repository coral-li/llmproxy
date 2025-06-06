#!/usr/bin/env python
"""Run LLMProxy tests"""

import sys
import pytest

if __name__ == "__main__":
    # Run pytest with appropriate arguments
    sys.exit(pytest.main([
        "llmproxy/tests/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-p", "no:warnings",  # Disable warnings
    ])) 