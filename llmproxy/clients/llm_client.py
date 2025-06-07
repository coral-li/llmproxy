import httpx
from typing import Dict, Any, AsyncIterator, Union, Optional
import json
from urllib.parse import urljoin
from pathlib import Path
import sys
import time

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified client for OpenAI and Azure OpenAI endpoints"""

    def __init__(self, timeout: float = 6000.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def close(self):
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
        is_azure = (
            endpoint_url.endswith("azure.com")
        )

        if is_azure:
            url = urljoin(endpoint_url + "/", "openai/v1/chat/completions")
        else:
            # Standard OpenAI URL
            url = urljoin(endpoint_url + "/", "v1/chat/completions")

        logger.debug(
            "llm_request", url=url, model=request_data.get("model"), stream=stream
        )

        if stream:
            return self._stream_request(url, headers, params, request_data)
        else:
            return await self._request(url, headers, params, request_data)

    async def _request(self, url: str, headers: dict, params: dict, data: dict) -> dict:
        """Make non-streaming request"""
        request_start_time = time.time()
        
        try:
            # Log full request details for debugging (with sanitized headers)
            sanitized_headers = {k: v if k.lower() != 'authorization' else 'Bearer <REDACTED>' for k, v in headers.items()}
            logger.debug(
                "llm_request_details",
                url=url,
                headers=sanitized_headers,
                params=params,
                request_body_sample=str(data)[:500] if data else None,  # First 500 chars of request
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
                
                # Enhanced error logging for debugging
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
                
                # For 500 errors, log full details
                if response.status_code >= 500:
                    logger.error(
                        "llm_request_server_error",
                        **error_details,
                        request_body=data,  # Include full request for 500 errors
                    )
                # For vague 400 errors, log request details to help debugging
                elif response.status_code == 400 and error and ("check your inputs" in error.lower() or "invalid_request_error" in error):
                    # Extract key request details for debugging
                    debug_details = {
                        **error_details,
                        "request_id": response.headers.get("x-request-id"),
                    }
                    
                    # Log prompt/messages content for debugging
                    if data:
                        # For chat completions
                        if "messages" in data:
                            messages_summary = []
                            for msg in data.get("messages", [])[:5]:  # First 5 messages
                                msg_summary = {
                                    "role": msg.get("role"),
                                    "content_preview": str(msg.get("content", ""))[:200] + "..." if len(str(msg.get("content", ""))) > 200 else str(msg.get("content", ""))
                                }
                                messages_summary.append(msg_summary)
                            debug_details["messages_preview"] = messages_summary
                            debug_details["total_messages"] = len(data.get("messages", []))
                        
                        # For responses API
                        if "input" in data:
                            input_data = data.get("input", [])
                            if isinstance(input_data, list):
                                input_summary = []
                                for inp in input_data[:3]:  # First 3 inputs
                                    if isinstance(inp, dict):
                                        inp_summary = {
                                            "role": inp.get("role"),
                                            "content_types": [c.get("type") for c in inp.get("content", [])] if isinstance(inp.get("content"), list) else None,
                                            "content_preview": str(inp)[:200] + "..." if len(str(inp)) > 200 else str(inp)
                                        }
                                        input_summary.append(inp_summary)
                                debug_details["input_preview"] = input_summary
                                debug_details["total_inputs"] = len(input_data)
                        
                        # Log other important parameters
                        debug_details["parameters"] = {
                            "temperature": data.get("temperature"),
                            "max_tokens": data.get("max_tokens"),
                            "max_output_tokens": data.get("max_output_tokens"),
                            "max_completion_tokens": data.get("max_completion_tokens"),
                            "stream": data.get("stream"),
                            "tools": len(data.get("tools", [])) if "tools" in data else None,
                            "tool_choice": data.get("tool_choice"),
                        }
                    
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

        except httpx.TimeoutException as e:
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

    async def _stream_request(
        self, url: str, headers: dict, params: dict, data: dict
    ) -> AsyncIterator[str]:
        """Make streaming request"""
        import time
        request_start_time = time.time()
        
        try:
            # Log request details for debugging
            sanitized_headers = {k: v if k.lower() != 'authorization' else 'Bearer <REDACTED>' for k, v in headers.items()}
            logger.debug(
                "llm_stream_request_start",
                url=url,
                headers=sanitized_headers,
                params=params,
                model=data.get("model") if data else None,
            )
            
            async with self.client.stream(
                "POST", url, headers=headers, params=params, json=data
            ) as response:
                # For streaming, we need to check status before yielding
                if response.status_code != 200:
                    error_text = await response.aread()
                    duration_ms = int((time.time() - request_start_time) * 1000)
                    
                    # Enhanced error logging for streaming
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
                    
                    if response.status_code >= 500:
                        logger.error(
                            "llm_stream_server_error",
                            **error_details,
                            request_body=data,  # Include full request for 500 errors
                        )
                    # For vague 400 errors in streaming
                    elif response.status_code == 400 and ("check your inputs" in error_text.decode().lower() or "invalid_request_error" in error_text.decode()):
                        # Add debugging info similar to non-streaming
                        debug_details = {
                            **error_details,
                            "request_id": response.headers.get("x-request-id"),
                        }
                        
                        if data and "messages" in data:
                            messages_summary = []
                            for msg in data.get("messages", [])[:5]:
                                msg_summary = {
                                    "role": msg.get("role"),
                                    "content_preview": str(msg.get("content", ""))[:200] + "..." if len(str(msg.get("content", ""))) > 200 else str(msg.get("content", ""))
                                }
                                messages_summary.append(msg_summary)
                            debug_details["messages_preview"] = messages_summary
                            debug_details["total_messages"] = len(data.get("messages", []))
                        
                        logger.error(
                            "llm_stream_vague_error",
                            **debug_details,
                        )
                    else:
                        logger.error(
                            "llm_stream_error",
                            **error_details,
                        )

                    # Yield error in SSE format
                    error_data = {
                        "error": {
                            "message": error_text.decode(),
                            "type": "api_error",
                            "code": response.status_code,
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Stream the response - filter out problematic chunks
                async for line in response.aiter_lines():
                    if line and line.startswith('data: '):
                        # Skip the [DONE] marker
                        if line == 'data: [DONE]':
                            yield line + "\n\n"
                            continue
                            
                        # Try to parse the data to filter out empty chunks
                        try:
                            data_str = line[6:]  # Remove 'data: ' prefix
                            chunk_data = json.loads(data_str)
                            
                            # Filter out chunks with empty choices arrays
                            if 'choices' in chunk_data and len(chunk_data['choices']) == 0:
                                logger.debug("Filtering out chunk with empty choices array")
                                continue
                            
                            # Filter out chunks with no meaningful content
                            # (e.g., just role assignment without content)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice:
                                    delta = choice['delta']
                                    # Skip if delta only contains role without content
                                    if 'role' in delta and 'content' not in delta:
                                        logger.debug("Filtering out chunk with only role assignment")
                                        continue
                                    # Don't filter out whitespace content - it's legitimate
                                    # Only skip if content is exactly empty string ""
                                    # Whitespace like "\n" or "  " should be kept
                            
                            # This chunk looks good, yield it
                            yield line + "\n\n"
                            
                        except json.JSONDecodeError:
                            # If we can't parse it, just forward it as-is
                            logger.debug(f"Could not parse chunk, forwarding as-is: {line}")
                            yield line + "\n\n"
                    elif line:
                        # Non-SSE formatted line, forward as-is
                        yield line + "\n\n"

        except httpx.TimeoutException:
            duration_ms = int((time.time() - request_start_time) * 1000)
            logger.error(
                "llm_stream_timeout", 
                url=url, 
                timeout=self.timeout,
                duration_ms=duration_ms,
                request_model=data.get("model") if data else None,
            )
            error_data = {
                "error": {
                    "message": f"Request timeout after {self.timeout}s",
                    "type": "timeout_error",
                    "code": 504,
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

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
            error_data = {
                "error": {"message": str(e), "type": "client_error", "code": 500}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

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
        is_azure = (
            "azure.com" in endpoint_url
        )

        if is_azure:
            # Azure OpenAI may have different URL pattern for responses
            url = urljoin(endpoint_url + "/", "openai/v1/responses")
        else:
            # Standard OpenAI URL for responses
            url = urljoin(endpoint_url + "/", "v1/responses")

        logger.debug(
            "llm_response_request", url=url, model=request_data.get("model"), stream=stream
        )

        if stream:
            return self._stream_response_request(url, headers, params, request_data)
        else:
            return await self._request(url, headers, params, request_data)

    async def _stream_response_request(
        self, url: str, headers: dict, params: dict, data: dict
    ) -> AsyncIterator[str]:
        """Make streaming request for Responses API"""
        try:
            async with self.client.stream(
                "POST", url, headers=headers, params=params, json=data
            ) as response:
                # For streaming, we need to check status before yielding
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(
                        "llm_response_stream_error",
                        status_code=response.status_code,
                        error=error_text.decode()[:500],
                    )

                    # Yield error in SSE format
                    error_data = {
                        "error": {
                            "message": error_text.decode(),
                            "type": "api_error",
                            "code": response.status_code,
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Stream the response - Forward the raw SSE stream
                # The httpx response already provides proper SSE formatting
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk.decode('utf-8')

        except httpx.TimeoutException:
            logger.error("llm_response_stream_timeout", url=url, timeout=self.timeout)
            error_data = {
                "error": {
                    "message": f"Request timeout after {self.timeout}s",
                    "type": "timeout_error",
                    "code": 504,
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("llm_response_stream_exception", url=url, error=str(e))
            error_data = {
                "error": {"message": str(e), "type": "client_error", "code": 500}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
