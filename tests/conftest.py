import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Generator, Optional

# Add the project root to Python path before other imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
import requests  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402

from llmproxy.main import app  # noqa: E402


def find_free_port() -> int:
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class MockOpenAIServer:
    """Simple mock OpenAI API server for testing"""

    def __init__(self, port: int):
        self.port = port
        self.host = "127.0.0.1"
        self.app = FastAPI()
        self._setup_routes()
        self.server_thread: Optional[threading.Thread] = None
        self.server: Optional[uvicorn.Server] = None

    def _setup_routes(self):
        @self.app.post("/chat/completions")
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._handle_chat_request(request)

        @self.app.post("/responses")
        @self.app.post("/v1/responses")
        async def responses(request: Request):
            return await self._handle_chat_request(request)

    async def _handle_chat_request(self, request: Request):
        body = await request.json()

        is_streaming = body.get("stream", False)
        model = body.get("model", "gpt-3.5-turbo")

        user_message, is_responses_api = self._extract_user_message(body, request)
        response_content = self._select_response_content(user_message)

        if is_responses_api:
            if is_streaming:
                return StreamingResponse(
                    self._stream_responses_api(response_content, model),
                    media_type="text/plain",
                )
            else:
                return {
                    "id": "resp-mock123",
                    "object": "response",
                    "created": int(time.time()),
                    "model": model,
                    "output": [
                        {
                            "id": "msg-mock123",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": response_content}
                            ],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": len(response_content.split()),
                        "total_tokens": 10 + len(response_content.split()),
                    },
                }
        else:
            if is_streaming:
                return StreamingResponse(
                    self._stream_response(response_content, model),
                    media_type="text/plain",
                )
            else:
                return {
                    "id": "chatcmpl-mock123",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_content,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": len(response_content.split()),
                        "total_tokens": 10 + len(response_content.split()),
                    },
                }

    def _extract_user_message(self, body, request):
        is_responses_api = "/responses" in str(request.url)
        if is_responses_api:
            input_text = body.get("input", "")
            instructions = body.get("instructions", "")
            return f"{instructions} {input_text}".strip(), True
        else:
            messages = body.get("messages", [])
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            return user_message, False

    def _select_response_content(self, user_message):
        if "Hello from LLMProxy" in user_message:
            return "Hello from LLMProxy!"
        elif "Count from 1 to 5" in user_message:
            return "1\n2\n3\n4\n5"
        elif "2+2" in user_message:
            return "4"
        elif (
            "three colors" in user_message.lower()
            or "primary colors" in user_message.lower()
        ):
            return "Red\nBlue\nYellow"
        elif "capital of France" in user_message:
            return "Paris"
        elif (
            "bedtime story" in user_message.lower()
            and "unicorn" in user_message.lower()
        ):
            return "Once upon a time, a magical unicorn danced under the stars before falling asleep in a meadow of dreams."
        else:
            return f"Mock response to: {user_message}"

    async def _stream_response(self, content: str, model: str):
        """Generate streaming response"""
        # Add artificial delay for cache testing (simulate network latency for first request)
        # This ensures measurable timing differences between cached and non-cached requests
        await asyncio.sleep(0.1)

        # First chunk
        chunk = {
            "id": "chatcmpl-mock123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Content chunks
        for word in content.split():
            chunk = {
                "id": "chatcmpl-mock123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"{word} "},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

        # Final chunk
        chunk = {
            "id": "chatcmpl-mock123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_responses_api(self, content: str, model: str):
        """Generate streaming response for responses API"""
        # Add artificial delay for cache testing (simulate network latency for first request)
        # This ensures measurable timing differences between cached and non-cached requests
        await asyncio.sleep(0.1)

        # Initial event
        event = {
            "type": "response.created",
            "response": {
                "id": "resp-mock123",
                "object": "response",
                "created": int(time.time()),
                "model": model,
                "status": "in_progress",
            },
        }
        yield "event: response.created\n"
        yield f"data: {json.dumps(event)}\n\n"

        # Content delta events
        for word in content.split():
            event = {"type": "response.output_text.delta", "delta": f"{word} "}
            yield "event: response.output_text.delta\n"
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.01)

        # Completion event
        event = {
            "type": "response.done",
            "response": {
                "id": "resp-mock123",
                "object": "response",
                "created": int(time.time()),
                "model": model,
                "status": "completed",
                "output": [{"type": "text", "text": content}],
            },
        }
        yield "event: response.done\n"
        yield f"data: {json.dumps(event)}\n\n"
        yield "event: done\n"
        yield "data: [DONE]\n\n"

    def start(self):
        """Start the mock server in a thread"""

        def run_server():
            config = uvicorn.Config(
                self.app, host=self.host, port=self.port, log_level="error"
            )
            self.server = uvicorn.Server(config)
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.get_event_loop().run_until_complete(self.server.serve())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to be ready
        self._wait_for_server()

    def stop(self):
        """Stop the mock server"""
        if self.server:
            self.server.should_exit = True

    def _wait_for_server(self, timeout: int = 10):
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                requests.post(
                    f"http://{self.host}:{self.port}/chat/completions",
                    json={"model": "test", "messages": []},
                    timeout=1.0,
                )
                return
            except Exception:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"Mock server failed to start within {timeout} seconds")


class LLMProxyTestServer:
    """Test server manager for llmproxy"""

    def __init__(self):
        self.host = "127.0.0.1"
        self.port = find_free_port()
        self.mock_servers = []
        self.server_thread: Optional[threading.Thread] = None
        self.server: Optional[uvicorn.Server] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Start the llmproxy server and mock backends"""
        # Check if Redis is available first
        self._check_redis()

        # Start mock servers first
        mock_ports = [8001, 8002, 8003]
        for port in mock_ports:
            mock_server = MockOpenAIServer(port)
            mock_server.start()
            self.mock_servers.append(mock_server)
            print(f"Started mock server on port {port}")

        # Set environment variable for config path
        os.environ["LLMPROXY_CONFIG"] = "llmproxy.test.yaml"

        # Start llmproxy server in a thread
        def run_server():
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level="error",
                access_log=False,
                server_header=False,
            )
            self.server = uvicorn.Server(config)
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.get_event_loop().run_until_complete(self.server.serve())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to be ready
        self._wait_for_server()

    def stop(self) -> None:
        """Stop the llmproxy server and mock backends"""
        # Stop llmproxy server
        if self.server:
            self.server.should_exit = True

        # Stop mock servers
        for mock_server in self.mock_servers:
            mock_server.stop()
        self.mock_servers.clear()

        # Clean up environment
        if "LLMPROXY_CONFIG" in os.environ:
            del os.environ["LLMPROXY_CONFIG"]

    def _check_redis(self):
        """Check if Redis is available"""
        import redis

        try:
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()
            print("✅ Redis is available")
        except Exception as e:
            raise RuntimeError(
                f"❌ Redis is not available: {e}\n"
                "Please start Redis before running tests. For example:\n"
                "  brew services start redis  # on macOS\n"
                "  sudo systemctl start redis  # on Linux\n"
                "  redis-server  # manually"
            )

    def _wait_for_server(self, timeout: int = 30) -> None:
        """Wait for the server to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.url}/health", timeout=2.0)
                if response.status_code == 200:
                    print(f"✅ LLMProxy server ready at {self.url}")
                    return
            except Exception:
                pass

            time.sleep(0.5)

        raise RuntimeError(f"Server failed to start within {timeout} seconds")


# Global test server instance
_test_server: Optional[LLMProxyTestServer] = None


@pytest.fixture(scope="session")
def llmproxy_server() -> Generator[LLMProxyTestServer, None, None]:
    """Start llmproxy server for testing"""
    global _test_server

    if _test_server is None:
        _test_server = LLMProxyTestServer()
        _test_server.start()
        print(f"\n=== Started LLMProxy test server at {_test_server.url} ===")

    try:
        yield _test_server
    finally:
        if _test_server is not None:
            _test_server.stop()
            print("\n=== Stopped LLMProxy test server ===")
            _test_server = None


@pytest.fixture(scope="session")
def proxy_url(llmproxy_server: LLMProxyTestServer) -> str:
    """Get the proxy URL for testing"""
    return llmproxy_server.url


@pytest.fixture(scope="session")
def openai_client(proxy_url: str) -> OpenAI:
    """Create OpenAI client pointing to test proxy"""
    return OpenAI(base_url=proxy_url, api_key="test-key")  # Dummy key for testing


@pytest.fixture(scope="session")
def async_openai_client(proxy_url: str) -> AsyncOpenAI:
    """Create AsyncOpenAI client pointing to test proxy"""
    return AsyncOpenAI(base_url=proxy_url, api_key="test-key")  # Dummy key for testing


@pytest.fixture(scope="session")
def model() -> str:
    """Get a configured model for testing"""
    return "gpt-3.5-turbo"


@pytest.fixture(scope="session", autouse=True)
def clear_cache(proxy_url: str):
    """Clear all caches before starting tests"""
    print("Clearing cache before tests...")
    try:
        response = requests.delete(f"{proxy_url}/cache", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            deleted_count = data.get("deleted_entries", 0)
            print(f"✅ Cache cleared: {deleted_count} entries deleted")
        else:
            print(f"⚠️  Cache clear returned status {response.status_code}")
    except Exception as e:
        print(f"⚠️  Cache clear failed: {str(e)}")


@pytest.fixture(autouse=True)
def test_setup():
    """Setup for each individual test"""
    # This runs before each test
    yield
    # This runs after each test (cleanup if needed)
    pass
