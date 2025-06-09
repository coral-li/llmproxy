import json
import time
from typing import AsyncIterator, Optional, Union
from urllib.parse import urljoin

import httpx

from llmproxy.core.logger import get_logger

# Imports are handled properly through package structure


logger = get_logger(__name__)


class LLMClient:
    """Unified client for OpenAI and Azure OpenAI endpoints"""

    def __init__(self, timeout: float = 6000.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def close(self) -> None:
        """Close the HTTP client"""
        await self.client.aclose()

    async def create_chat_completion(
        self,
        model: str,
        endpoint_url: str,
        api_key: str,
        request_data: dict,
        default_query: Optional[dict] = None,
        stream: bool = False,
    ) -> Union[dict, AsyncIterator[str]]:
        """Make chat completion request to OpenAI-compatible endpoint"""

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Handle Azure OpenAI query parameters
        params = default_query or {}

        # Build full URL
        if endpoint_url.endswith("/"):
            endpoint_url = endpoint_url[:-1]

        # Detect Azure OpenAI endpoints
        is_azure = endpoint_url.endswith("azure.com")

        if is_azure:
            url = urljoin(endpoint_url + "/", "openai/v1/chat/completions")
        else:
            # Standard OpenAI URL
            url = urljoin(endpoint_url + "/", "v1/chat/completions")

        logger.debug(
            "llm_request", url=url, model=request_data.get("model"), stream=stream
        )

        if stream:
            return await self._stream_request(url, headers, params, request_data)
        else:
            return await self._request(url, headers, params, request_data)

    async def _request(self, url: str, headers: dict, params: dict, data: dict) -> dict:
        """Make non-streaming request"""
        request_start_time = time.time()

        try:
            # Log full request details for debugging (with sanitized headers)
            sanitized_headers = {
                k: v if k.lower() != "authorization" else "Bearer <REDACTED>"
                for k, v in headers.items()
            }
            logger.debug(
                "llm_request_details",
                url=url,
                headers=sanitized_headers,
                params=params,
                request_body_sample=(
                    str(data)[:500] if data else None
                ),  # First 500 chars of request
                model=data.get("model") if data else None,
            )

            response = await self.client.post(
                url, headers=headers, params=params, json=data
            )

            # Calculate request duration
            request_duration_ms = int((time.time() - request_start_time) * 1000)

            # Parse response
            response_data = None
            error = None

            if response.status_code == 200:
                response_data = response.json()
                logger.debug(
                    "llm_request_success",
                    url=url,
                    status_code=response.status_code,
                    duration_ms=request_duration_ms,
                )
            else:
                error = response.text
                error_details = self._log_error_details(
                    response, url, data, request_duration_ms
                )
                # For 500 errors, log full details
                if response.status_code >= 500:
                    logger.error(
                        "llm_request_server_error",
                        **error_details,
                        request_body=data,  # Include full request for 500 errors
                    )
                # For vague 400 errors, log request details to help debugging
                elif (
                    response.status_code == 400
                    and error
                    and (
                        "check your inputs" in error.lower()
                        or "invalid_request_error" in error
                    )
                ):
                    debug_details = self._log_debug_details(
                        response, data, error_details
                    )
                    logger.error(
                        "llm_request_vague_error",
                        **debug_details,
                    )
                else:
                    logger.error(
                        "llm_request_error",
                        **error_details,
                    )

            # Return response with headers for rate limit parsing
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": response_data,
                "error": error,
                "duration_ms": request_duration_ms,
            }

        except httpx.TimeoutException:
            duration_ms = int((time.time() - request_start_time) * 1000)
            logger.error(
                "llm_request_timeout",
                url=url,
                timeout=self.timeout,
                duration_ms=duration_ms,
                request_model=data.get("model") if data else None,
            )
            return {
                "status_code": 504,
                "headers": {},
                "data": None,
                "error": f"Request timeout after {self.timeout}s",
                "duration_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = int((time.time() - request_start_time) * 1000)
            logger.error(
                "llm_request_exception",
                url=url,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                request_model=data.get("model") if data else None,
            )
            return {
                "status_code": 500,
                "headers": {},
                "data": None,
                "error": str(e),
                "duration_ms": duration_ms,
            }

    def _log_error_details(
        self,
        response: httpx.Response,
        url: str,
        data: Optional[dict],
        request_duration_ms: float,
    ) -> dict:
        error = response.text
        error_details = {
            "status_code": response.status_code,
            "url": url,
            "method": "POST",
            "duration_ms": request_duration_ms,
            "response_headers": dict(response.headers),
            "error_body": error,
            "request_model": data.get("model") if data else None,
            "request_size": len(json.dumps(data)) if data else 0,
        }
        return error_details

    def _log_debug_details(
        self, response: httpx.Response, data: Optional[dict], error_details: dict
    ) -> dict:
        debug_details = {
            **error_details,
            "request_id": response.headers.get("x-request-id"),
        }
        if data:
            if "messages" in data:
                messages_summary = []
                for msg in data.get("messages", [])[:5]:
                    msg_summary = {
                        "role": msg.get("role"),
                        "content_preview": (
                            str(msg.get("content", ""))[:200] + "..."
                            if len(str(msg.get("content", ""))) > 200
                            else str(msg.get("content", ""))
                        ),
                    }
                    messages_summary.append(msg_summary)
                debug_details["messages_preview"] = messages_summary
                debug_details["total_messages"] = len(data.get("messages", []))
            if "input" in data:
                input_data = data.get("input", [])
                if isinstance(input_data, list):
                    input_summary = []
                    for inp in input_data[:3]:
                        if isinstance(inp, dict):
                            inp_summary = {
                                "role": inp.get("role"),
                                "content_types": (
                                    [c.get("type") for c in inp.get("content", [])]
                                    if isinstance(inp.get("content"), list)
                                    else None
                                ),
                                "content_preview": (
                                    str(inp)[:200] + "..."
                                    if len(str(inp)) > 200
                                    else str(inp)
                                ),
                            }
                            input_summary.append(inp_summary)
                    debug_details["input_preview"] = input_summary
                    debug_details["total_inputs"] = len(input_data)
            debug_details["parameters"] = {
                "temperature": data.get("temperature"),
                "max_tokens": data.get("max_tokens"),
                "max_output_tokens": data.get("max_output_tokens"),
                "max_completion_tokens": data.get("max_completion_tokens"),
                "stream": data.get("stream"),
                "tools": (len(data.get("tools", [])) if "tools" in data else None),
                "tool_choice": data.get("tool_choice"),
            }
        return debug_details

    async def _stream_request(
        self, url: str, headers: dict, params: dict, data: dict
    ) -> Union[dict, AsyncIterator[str]]:
        """Make streaming request"""
        import time

        request_start_time = time.time()

        # Log request details for debugging
        sanitized_headers = {
            k: v if k.lower() != "authorization" else "Bearer <REDACTED>"
            for k, v in headers.items()
        }
        logger.debug(
            "llm_stream_request_start",
            url=url,
            headers=sanitized_headers,
            params=params,
            model=data.get("model") if data else None,
        )

        try:
            response = self.client.stream(
                "POST", url, headers=headers, params=params, json=data
            )

            # Check if it's an error by peeking at the response
            # We'll handle the actual streaming in a wrapper
            async with response as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    error_details, duration_ms = self._handle_stream_error(
                        resp, error_text, data, request_start_time, url
                    )
                    if resp.status_code >= 500:
                        logger.error(
                            "llm_stream_server_error",
                            **error_details,
                            request_body=data,  # Include full request for 500 errors
                        )
                    elif resp.status_code == 400 and (
                        "check your inputs" in error_text.decode().lower()
                        or "invalid_request_error" in error_text.decode()
                    ):
                        debug_details = {
                            **error_details,
                            "request_id": resp.headers.get("x-request-id"),
                        }
                        if data and "messages" in data:
                            messages_summary = []
                            for msg in data.get("messages", [])[:5]:
                                msg_summary = {
                                    "role": msg.get("role"),
                                    "content_preview": (
                                        str(msg.get("content", ""))[:200] + "..."
                                        if len(str(msg.get("content", ""))) > 200
                                        else str(msg.get("content", ""))
                                    ),
                                }
                                messages_summary.append(msg_summary)
                            debug_details["messages_preview"] = messages_summary
                            debug_details["total_messages"] = len(
                                data.get("messages", [])
                            )
                        logger.error(
                            "llm_stream_vague_error",
                            **debug_details,
                        )
                    else:
                        logger.error(
                            "llm_stream_error",
                            **error_details,
                        )

                    # Return error dict instead of yielding error chunks
                    return {
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "data": None,
                        "error": error_text.decode(),
                        "duration_ms": duration_ms,
                    }

            # Success - return a generator that manages its own context
            async def stream_response() -> AsyncIterator[str]:
                async with self.client.stream(
                    "POST", url, headers=headers, params=params, json=data
                ) as response:
                    async for line in response.aiter_lines():
                        for chunk in self._filter_and_yield_chunk(line):
                            yield chunk

            return stream_response()

        except httpx.TimeoutException:
            duration_ms = int((time.time() - request_start_time) * 1000)
            logger.error(
                "llm_stream_timeout",
                url=url,
                timeout=self.timeout,
                duration_ms=duration_ms,
                request_model=data.get("model") if data else None,
            )
            # Return error dict for timeout
            return {
                "status_code": 504,
                "headers": {},
                "data": None,
                "error": f"Request timeout after {self.timeout}s",
                "duration_ms": duration_ms,
            }

        except Exception as e:
            duration_ms = int((time.time() - request_start_time) * 1000)
            logger.error(
                "llm_stream_exception",
                url=url,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                request_model=data.get("model") if data else None,
            )
            # Return error dict for exceptions
            return {
                "status_code": 500,
                "headers": {},
                "data": None,
                "error": str(e),
                "duration_ms": duration_ms,
            }

    def _handle_stream_error(
        self,
        response: httpx.Response,
        error_text: bytes,
        data: Optional[dict],
        request_start_time: float,
        url: str,
    ) -> tuple:
        duration_ms = int((time.time() - request_start_time) * 1000)
        error_details = {
            "status_code": response.status_code,
            "url": url,
            "method": "POST",
            "duration_ms": duration_ms,
            "response_headers": dict(response.headers),
            "error_body": error_text.decode(),
            "request_model": data.get("model") if data else None,
            "streaming": True,
        }
        return error_details, duration_ms

    def _filter_and_yield_chunk(self, line: str) -> list:
        if line and line.startswith("data: "):
            if line == "data: [DONE]":
                return [line + "\n\n"]
            try:
                data_str = line[6:]
                chunk_data = json.loads(data_str)
                if "choices" in chunk_data and len(chunk_data["choices"]) == 0:
                    logger.debug("Filtering out chunk with empty choices array")
                    return []
                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                    choice = chunk_data["choices"][0]
                    if "delta" in choice:
                        delta = choice["delta"]
                        if "role" in delta and "content" not in delta:
                            logger.debug(
                                "Filtering out chunk with only role assignment"
                            )
                            return []
                return [line + "\n\n"]
            except json.JSONDecodeError:
                logger.debug(f"Could not parse chunk, forwarding as-is: {line}")
                return [line + "\n\n"]
        elif line:
            return [line + "\n\n"]
        return []

    async def create_response(
        self,
        model: str,
        endpoint_url: str,
        api_key: str,
        request_data: dict,
        default_query: Optional[dict] = None,
        stream: bool = False,
    ) -> Union[dict, AsyncIterator[str]]:
        """Make response API request to OpenAI-compatible endpoint"""

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Handle Azure OpenAI query parameters
        params = default_query or {}

        # Build full URL
        if endpoint_url.endswith("/"):
            endpoint_url = endpoint_url[:-1]

        # Detect Azure OpenAI endpoints
        is_azure = "azure.com" in endpoint_url

        if is_azure:
            # Azure OpenAI may have different URL pattern for responses
            url = urljoin(endpoint_url + "/", "openai/v1/responses")
        else:
            # Standard OpenAI URL for responses
            url = urljoin(endpoint_url + "/", "v1/responses")

        logger.debug(
            "llm_response_request",
            url=url,
            model=request_data.get("model"),
            stream=stream,
        )

        if stream:
            return await self._stream_response_request(
                url, headers, params, request_data
            )
        else:
            return await self._request(url, headers, params, request_data)

    async def _stream_response_request(
        self, url: str, headers: dict, params: dict, data: dict
    ) -> Union[dict, AsyncIterator[str]]:
        """Make streaming request for Responses API"""
        import time

        request_start_time = time.time()

        try:
            response = self.client.stream(
                "POST", url, headers=headers, params=params, json=data
            )

            # Check if it's an error by peeking at the response
            async with response as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    duration_ms = int((time.time() - request_start_time) * 1000)
                    logger.error(
                        "llm_response_stream_error",
                        status_code=resp.status_code,
                        error=error_text.decode()[:500],
                        duration_ms=duration_ms,
                    )

                    # Return error dict instead of yielding error chunks
                    return {
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "data": None,
                        "error": error_text.decode(),
                        "duration_ms": duration_ms,
                    }

            # Success - return a generator that manages its own context
            async def stream_response() -> AsyncIterator[str]:
                async with self.client.stream(
                    "POST", url, headers=headers, params=params, json=data
                ) as response:
                    # Stream the response line by line for proper event parsing
                    # The responses API uses event-based SSE format that needs line-by-line processing
                    async for line in response.aiter_lines():
                        # Yield each line with proper newline formatting
                        if line:
                            # Non-empty lines get a single newline
                            yield line + "\n"
                        else:
                            # Empty lines indicate event boundary, add another newline
                            yield "\n"

                        # Log event lines for debugging
                        if line.startswith("event: "):
                            logger.debug("response_api_event_line", event_line=line)

            return stream_response()

        except httpx.TimeoutException:
            duration_ms = int((time.time() - request_start_time) * 1000)
            logger.error(
                "llm_response_stream_timeout",
                url=url,
                timeout=self.timeout,
                duration_ms=duration_ms,
            )
            # Return error dict for timeout
            return {
                "status_code": 504,
                "headers": {},
                "data": None,
                "error": f"Request timeout after {self.timeout}s",
                "duration_ms": duration_ms,
            }

        except Exception as e:
            duration_ms = int((time.time() - request_start_time) * 1000)
            logger.error(
                "llm_response_stream_exception",
                url=url,
                error=str(e),
                duration_ms=duration_ms,
            )
            # Return error dict for exceptions
            return {
                "status_code": 500,
                "headers": {},
                "data": None,
                "error": str(e),
                "duration_ms": duration_ms,
            }
