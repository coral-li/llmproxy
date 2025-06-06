import httpx
from typing import Dict, Any, AsyncIterator, Union, Optional
import json
from urllib.parse import urljoin
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified client for OpenAI and Azure OpenAI endpoints"""

    def __init__(self, timeout: float = 60.0):
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
            "azure.com" in endpoint_url
        )

        if is_azure:
            # Azure URL without deployment - try to extract from endpoint URL
            if "/openai/deployments/" in endpoint_url:
                # Deployment already in URL
                url = f"{endpoint_url}/chat/completions"
            else:
                # Correct URL format: /openai/deployments/gpt-4.1/chat/completions
                url = f"{endpoint_url}/openai/v1/chat/completions"
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
        try:
            response = await self.client.post(
                url, headers=headers, params=params, json=data
            )

            # Parse response
            response_data = None
            error = None

            if response.status_code == 200:
                response_data = response.json()
            else:
                error = response.text
                logger.error(
                    "llm_request_error",
                    status_code=response.status_code,
                    error=error[:500],
                )  # Truncate long errors

            # Return response with headers for rate limit parsing
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": response_data,
                "error": error,
            }

        except httpx.TimeoutException as e:
            logger.error("llm_request_timeout", url=url, timeout=self.timeout)
            return {
                "status_code": 504,
                "headers": {},
                "data": None,
                "error": f"Request timeout after {self.timeout}s",
            }
        except Exception as e:
            logger.error("llm_request_exception", url=url, error=str(e))
            return {"status_code": 500, "headers": {}, "data": None, "error": str(e)}

    async def _stream_request(
        self, url: str, headers: dict, params: dict, data: dict
    ) -> AsyncIterator[str]:
        """Make streaming request"""
        try:
            async with self.client.stream(
                "POST", url, headers=headers, params=params, json=data
            ) as response:
                # For streaming, we need to check status before yielding
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(
                        "llm_stream_error",
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

                # Stream the response
                async for line in response.aiter_lines():
                    if line:
                        yield line + "\n"

        except httpx.TimeoutException:
            logger.error("llm_stream_timeout", url=url, timeout=self.timeout)
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
            logger.error("llm_stream_exception", url=url, error=str(e))
            error_data = {
                "error": {"message": str(e), "type": "client_error", "code": 500}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
