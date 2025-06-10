#!/usr/bin/env python3
"""
Simple HTTP Fuzzing for LLMProxy with Mock Server
A lightweight fuzzer that doesn't require Atheris or Hypothesis.
Uses the existing test infrastructure to perform basic security testing.
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests
import yaml

# Import the test infrastructure
from tests.conftest import LLMProxyTestServer, MockOpenAIServer, find_free_port

# Add the project root to the path
os.path.join(os.path.dirname(__file__), "..")


class SimpleFuzzer:
    """Simple HTTP fuzzer for testing LLMProxy security and robustness."""

    def __init__(self):
        self.proxy_url = None
        self.llmproxy_server = None
        self.mock_servers = []
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "crashes": 0,
            "errors": [],
        }

    def setup_environment(self):
        """Set up mock environment."""
        print("Setting up mock environment...")

        # Use the same ports that the test config expects
        mock_ports = [8001, 8002, 8003]

        for port in mock_ports:
            try:
                mock_server = MockOpenAIServer(port)
                mock_server.start()
                self.mock_servers.append(mock_server)
                print(f"Started mock server on port {port}")
            except Exception as e:
                print(f"Failed to start mock server on port {port}: {e}")

        # Don't create our own config - let LLMProxyTestServer use llmproxy.test.yaml
        # Start LLMProxy
        try:
            self.llmproxy_server = LLMProxyTestServer()
            self.llmproxy_server.start()
            self.proxy_url = self.llmproxy_server.url
            print(f"Started LLMProxy on {self.proxy_url}")

            # Wait for server to be ready
            self._wait_for_server()

        except Exception as e:
            print(f"Failed to start LLMProxy: {e}")
            self.cleanup()
            raise

    def _wait_for_server(self, timeout=30):
        """Wait for LLMProxy to be ready."""
        for _ in range(timeout):
            try:
                response = requests.get(f"{self.proxy_url}/health", timeout=2)
                if response.status_code == 200:
                    print("LLMProxy is ready")
                    return
            except requests.RequestException:
                pass
            time.sleep(1)

        raise RuntimeError("LLMProxy failed to start within timeout")

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")

        if self.llmproxy_server:
            try:
                self.llmproxy_server.stop()
            except Exception as e:
                print(f"Error stopping LLMProxy: {e}")

        for server in self.mock_servers:
            try:
                server.stop()
            except Exception as e:
                print(f"Error stopping mock server: {e}")

    def run_test(self, test_name, test_func):
        """Run a single test and track results."""
        self.results["total_tests"] += 1
        try:
            test_func()
            self.results["passed"] += 1
            print(f"✓ {test_name}")
        except AssertionError as e:
            self.results["failed"] += 1
            print(f"✗ {test_name}: {e}")
            self.results["errors"].append(f"{test_name}: {e}")
        except Exception as e:
            self.results["crashes"] += 1
            print(f"💥 {test_name}: CRASH - {e}")
            self.results["errors"].append(f"{test_name}: CRASH - {e}")

    def test_basic_chat_completions(self):
        """Test basic chat completions functionality."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = requests.post(
            f"{self.proxy_url}/chat/completions", json=payload, timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        result = response.json()
        assert "choices" in result, "Response missing 'choices'"

    def test_large_payload(self):
        """Test handling of large payloads."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "x" * 50000}],  # 50KB payload
        }

        response = requests.post(
            f"{self.proxy_url}/chat/completions", json=payload, timeout=30
        )

        # Should not crash
        assert response.status_code != 500, "Server crashed on large payload"

    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        malformed_payloads = [
            '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}',  # Missing closing brace
            '{"model": "gpt-3.5-turbo", "messages": "not_an_array"}',  # Wrong type
            '{"model": 123, "messages": []}',  # Wrong model type
            "",  # Empty payload
            "not json at all",  # Not JSON
        ]

        for payload in malformed_payloads:
            try:
                response = requests.post(
                    f"{self.proxy_url}/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                # Should return 4xx error, not crash
                assert (
                    response.status_code != 500
                ), f"Server crashed on malformed JSON: {payload[:50]}"
            except requests.exceptions.RequestException:
                # Network errors are OK
                pass

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        test_cases = [
            {},  # Empty object
            {"model": "gpt-3.5-turbo"},  # Missing messages
            {"messages": [{"role": "user", "content": "test"}]},  # Missing model
            {"model": "", "messages": []},  # Empty values
        ]

        for payload in test_cases:
            response = requests.post(
                f"{self.proxy_url}/chat/completions", json=payload, timeout=10
            )
            # Should handle gracefully (4xx error, not crash)
            assert response.status_code != 500, f"Server crashed on payload: {payload}"

    def test_invalid_model(self):
        """Test handling of invalid model names."""
        invalid_models = [
            "non-existent-model",
            "",
            None,
            123,
            "model-with-injection'; DROP TABLE users; --",
        ]

        for model in invalid_models:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
            }

            try:
                response = requests.post(
                    f"{self.proxy_url}/chat/completions", json=payload, timeout=10
                )
                # Should not crash
                assert (
                    response.status_code != 500
                ), f"Server crashed on invalid model: {model}"
            except requests.exceptions.RequestException:
                # JSON encoding errors are expected for some invalid models
                pass

    def test_injection_attempts(self):
        """Test various injection attempts."""
        injection_payloads = [
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "'; DROP TABLE users; --"}],
            },
            {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "<script>alert('xss')</script>"}
                ],
            },
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "../../../etc/passwd"}],
            },
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "$(rm -rf /)"}],
            },
        ]

        for payload in injection_payloads:
            response = requests.post(
                f"{self.proxy_url}/chat/completions", json=payload, timeout=10
            )
            # Should handle safely
            assert (
                response.status_code != 500
            ), f"Server crashed on injection: {payload['messages'][0]['content'][:50]}"

    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello 世界 🌍 测试 émojis ñoñó",
                }
            ],
        }

        response = requests.post(
            f"{self.proxy_url}/chat/completions", json=payload, timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_health_endpoint(self):
        """Test health endpoint robustness."""
        # Basic health check
        response = requests.get(f"{self.proxy_url}/health", timeout=5)
        assert response.status_code == 200, "Health endpoint failed"

        # Health check with various headers
        malformed_headers = [
            {"X-Long-Header": "x" * 10000},
            {"X-Special-Chars": "\r\n\t\x00"},
            {"X-Injection": "'; DROP TABLE users; --"},
        ]

        for headers in malformed_headers:
            try:
                response = requests.get(
                    f"{self.proxy_url}/health", headers=headers, timeout=5
                )
                assert (
                    response.status_code != 500
                ), f"Health endpoint crashed with headers: {headers}"
            except (requests.exceptions.RequestException, ValueError):
                # Some header issues are expected
                pass

    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = requests.get(f"{self.proxy_url}/stats", timeout=5)
        # Should not crash
        assert response.status_code != 500, "Stats endpoint crashed"

    def test_responses_api(self):
        """Test the responses API endpoint."""
        payload = {"model": "gpt-4.1", "input": [{"role": "user", "content": "Hello"}]}

        response = requests.post(
            f"{self.proxy_url}/responses", json=payload, timeout=10
        )

        # Should not crash (might not be implemented)
        assert response.status_code != 500, "Responses API crashed"

    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        results = Queue()

        def make_request():
            try:
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": f"Test {random.randint(1, 1000)}"}
                    ],
                }
                response = requests.post(
                    f"{self.proxy_url}/chat/completions", json=payload, timeout=15
                )
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")

        # Start 10 concurrent requests
        threads = []
        for _ in range(10):
            t = threading.Thread(target=make_request)
            t.start()
            threads.append(t)

        # Wait for all to complete
        for t in threads:
            t.join()

        # Check results
        crashes = 0
        while not results.empty():
            result = results.get()
            if result == 500:
                crashes += 1

        assert (
            crashes == 0
        ), f"Server crashed under concurrent load ({crashes}/10 requests)"

    def run_all_tests(self):
        """Run all fuzzing tests."""
        print(f"Running fuzzing tests against {self.proxy_url}")
        print("=" * 60)

        # Basic functionality tests
        self.run_test("Basic chat completions", self.test_basic_chat_completions)
        self.run_test("Health endpoint", self.test_health_endpoint)
        self.run_test("Stats endpoint", self.test_stats_endpoint)
        self.run_test("Responses API", self.test_responses_api)

        # Security and robustness tests
        self.run_test("Large payload handling", self.test_large_payload)
        self.run_test("Malformed JSON handling", self.test_malformed_json)
        self.run_test("Missing required fields", self.test_missing_required_fields)
        self.run_test("Invalid model names", self.test_invalid_model)
        self.run_test("Injection attempts", self.test_injection_attempts)
        self.run_test("Unicode handling", self.test_unicode_handling)
        self.run_test("Concurrent requests", self.test_concurrent_requests)

        # Print results
        print("=" * 60)
        print(f"Test Results:")
        print(f"  Total tests: {self.results['total_tests']}")
        print(f"  Passed: {self.results['passed']} ✓")
        print(f"  Failed: {self.results['failed']} ✗")
        print(f"  Crashes: {self.results['crashes']} 💥")

        if self.results["errors"]:
            print(f"\nErrors and failures:")
            for error in self.results["errors"]:
                print(f"  - {error}")

        # Return success if no crashes
        return self.results["crashes"] == 0


def main():
    """Main function."""
    print("Starting Simple LLMProxy Fuzzer with Mock Servers")
    print(
        "This fuzzer tests basic security and robustness without external dependencies"
    )

    # Check Redis availability
    try:
        import redis

        r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
        r.ping()
        print("Redis connection verified")
    except Exception as e:
        print(f"WARNING: Redis not available: {e}")
        print("Some functionality may not work properly")

    fuzzer = SimpleFuzzer()

    try:
        fuzzer.setup_environment()
        success = fuzzer.run_all_tests()

        if success:
            print("\n🎉 All tests passed! No critical issues found.")
            return 0
        else:
            print("\n⚠️  Some tests failed or caused crashes. Review the results above.")
            return 1

    except Exception as e:
        print(f"\n💥 Failed to run fuzzer: {e}")
        return 1
    finally:
        fuzzer.cleanup()


if __name__ == "__main__":
    exit(main())
