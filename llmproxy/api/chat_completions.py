from typing import Dict, Any, Optional
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import time
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from managers.load_balancer import LoadBalancer
from managers.rate_limit_manager import RateLimitManager
from core.cache_manager import CacheManager
from clients.llm_client import LLMClient
from models.endpoint import Endpoint
from api.error_handler import APIError, RateLimitError, is_retryable_error
from config_model import LLMProxyConfig
from core.logger import get_logger

logger = get_logger(__name__)


class ChatCompletionHandler:
    """Handles chat completion requests with load balancing, caching, and retries"""

    def __init__(
        self,
        load_balancer: LoadBalancer,
        rate_limit_manager: RateLimitManager,
        cache_manager: CacheManager,
        llm_client: LLMClient,
        config: LLMProxyConfig,
    ):
        self.load_balancer = load_balancer
        self.rate_limit_manager = rate_limit_manager
        self.cache_manager = cache_manager
        self.llm_client = llm_client
        self.config = config

    async def handle_request(self, request_data: dict) -> Any:
        """Handle incoming chat completion request"""
        start_time = time.time()

        # Extract model and determine model group
        model = request_data.get("model")
        if not model:
            raise HTTPException(400, "Model is required")

        # Map model to model group (for now, assume model == model_group)
        # In production, you might have a mapping like gpt-4-turbo -> gpt-4.1
        model_group = model

        # Check if model group exists
        if model_group not in self.load_balancer.get_model_groups():
            raise HTTPException(400, f"Model '{model}' not configured")

        is_streaming = request_data.get("stream", False)

        # Check cache first
        if self.config.general_settings.cache:
            cached = await self.cache_manager.get(request_data)
            if cached:
                latency_ms = int((time.time() - start_time) * 1000)
                if is_streaming:
                    endpoint_base = cached.get("meta", {}).get(
                        "endpoint_base_url", "https://api.openai.com"
                    )

                    cached_data = cached.get("data")
                    if cached.get("stream"):
                        chunks = cached_data
                    else:
                        content = (
                            cached_data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        model_name = cached_data.get("model", request_data.get("model"))
                        base_obj = {
                            "id": cached_data.get("id", "cached"),
                            "object": "chat.completion.chunk",
                            "created": cached_data.get("created", int(time.time())),
                            "model": model_name,
                        }
                        chunk_role = json.dumps({**base_obj, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                        chunk_content = json.dumps({**base_obj, "choices": [{"index": 0, "delta": {"content": content}}]})
                        chunk_end = json.dumps({**base_obj, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                        chunks = [
                            f"data: {chunk_role}\n\n",
                            f"data: {chunk_content}\n\n",
                            f"data: {chunk_end}\n\n",
                            "data: [DONE]\n\n",
                        ]

                    async def stream_cached():
                        for chunk in chunks:
                            yield chunk
                
                    return StreamingResponse(
                        stream_cached(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                            "X-Proxy-Endpoint-Base-Url": endpoint_base,
                            "X-Proxy-Cache-Hit": "true",
                            "X-Proxy-Latency-MS": str(latency_ms),
                        },
                    )
                else:
                    cached_data = cached.get("data")
                    cached_data["_proxy_cache_hit"] = True
                    cached_data["_proxy_latency_ms"] = latency_ms
                    return cached_data

        # Execute request with retries
        response = await self._execute_with_failover(model_group, request_data)

        # For streaming responses, return immediately
        if is_streaming and response.get("status_code") == 200 and hasattr(response.get("data"), "__aiter__"):
            headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Proxy-Endpoint-Base-Url": response.get("endpoint_base_url", "https://api.openai.com"),
                "X-Proxy-Cache-Hit": "false",
                "X-Proxy-Latency-MS": str(int((time.time() - start_time) * 1000)),
            }
            return StreamingResponse(response["data"], media_type="text/event-stream", headers=headers)

        # Cache successful non-streaming responses
        if (
            self.config.general_settings.cache
            and not is_streaming
            and response.get("status_code") == 200
        ):
            await self.cache_manager.set(request_data, response["data"], is_streaming=False)

        # Add proxy metadata
        if response.get("data"):
            response["data"]["_proxy_cache_hit"] = False
            response["data"]["_proxy_latency_ms"] = int(
                (time.time() - start_time) * 1000
            )
            # Add endpoint information if available
            if response.get("endpoint_base_url"):
                response["data"]["_proxy_endpoint_base_url"] = response["endpoint_base_url"]

        # Return appropriate response
        if is_streaming:
            # If we reach here, streaming failed
            raise HTTPException(
                status_code=response.get("status_code", 500),
                detail=response.get("error", "Streaming response error"),
            )
        else:
            if response["status_code"] == 200:
                return response["data"]
            else:
                raise HTTPException(
                    status_code=response["status_code"],
                    detail=response.get("error", "Unknown error"),
                )

    async def _execute_with_failover(
        self,
        model_group: str,
        request_data: dict,
        attempted_endpoints: Optional[set] = None,
    ) -> dict:
        """Execute request with automatic failover on errors"""

        attempted_endpoints = attempted_endpoints or set()
        is_streaming = request_data.get("stream", False)

        for attempt in range(self.config.general_settings.num_retries):
            # Select an endpoint
            endpoint = await self.load_balancer.select_endpoint(model_group)

            if not endpoint:
                logger.error("no_endpoint_available", model_group=model_group)
                return {
                    "status_code": 503,
                    "headers": {},
                    "data": None,
                    "error": "No available endpoints",
                }

            # Skip if we've already tried this endpoint
            if endpoint.id in attempted_endpoints:
                continue

            attempted_endpoints.add(endpoint.id)

            try:
                # Make the request
                response = await self._make_request(
                    endpoint, request_data, is_streaming
                )

                # Handle streaming differently
                if is_streaming and response.get("status_code") == 200:
                    gen = response["data"]
                    if self.config.general_settings.cache:
                        async def caching_gen():
                            chunks = []
                            async for chunk in gen:
                                chunks.append(chunk)
                                yield chunk
                            await self.cache_manager.set(
                                request_data,
                                chunks,
                                is_streaming=True,
                                metadata={"endpoint_base_url": endpoint.params.get("base_url", "https://api.openai.com")},
                            )
                        gen = caching_gen()

                    return {
                        "status_code": 200,
                        "data": gen,
                        "headers": {},
                        "endpoint_base_url": endpoint.params.get("base_url", "https://api.openai.com"),
                    }

                # Check response status
                if response["status_code"] == 200:
                    # Success!
                    self.load_balancer.record_success(endpoint)

                    # Update rate limits
                    await self.rate_limit_manager.update_from_headers(
                        endpoint.id, response["headers"]
                    )

                    # Add endpoint information to response
                    response["endpoint_base_url"] = endpoint.params.get("base_url", "https://api.openai.com")

                    return response

                # Handle errors - retry with different endpoint regardless of error type
                error_msg = response.get("error", "Unknown error")

                if response["status_code"] == 429:
                    # Rate limit hit - this shouldn't happen often due to pre-checks
                    logger.warning("rate_limit_hit", endpoint_id=endpoint.id)
                    self.load_balancer.record_failure(endpoint, "Rate limit exceeded")

                    # Update rate limits from headers
                    await self.rate_limit_manager.update_from_headers(
                        endpoint.id, response["headers"]
                    )

                elif is_retryable_error(response["status_code"]):
                    # Retryable error
                    logger.warning(
                        "retryable_error",
                        endpoint_id=endpoint.id,
                        status_code=response["status_code"],
                        error=error_msg[:200],
                    )

                    self.load_balancer.record_failure(endpoint, error_msg)

                else:
                    # Non-retryable error (like 400 Bad Request) - still retry with different endpoint
                    logger.warning(
                        "non_retryable_error_retrying",
                        endpoint_id=endpoint.id,
                        status_code=response["status_code"],
                        error=error_msg[:200],
                    )

                    # Mark endpoint as failed and continue to next endpoint
                    self.load_balancer.record_failure(endpoint, error_msg)

                # Store the last response to return if all endpoints fail
                last_response = response

            except Exception as e:
                logger.error("request_exception", endpoint_id=endpoint.id, error=str(e))

                self.load_balancer.record_failure(endpoint, str(e))

                # For streaming, return error in SSE format
                if is_streaming:

                    async def error_stream():
                        yield f'data: {{"error": {{"message": "{str(e)}", "type": "proxy_error"}}}}\n\n'
                        yield "data: [DONE]\n\n"

                    return {"status_code": 500, "data": error_stream()}

        # All retries exhausted - return the last response we received
        # If we have a last response, return that; otherwise return a generic error
        if 'last_response' in locals():
            return last_response
        else:
            return {
                "status_code": 503,
                "headers": {},
                "data": None,
                "error": "All endpoints failed after retries",
            }

    def _filter_proxy_params(self, request_data: dict) -> dict:
        """Filter out proxy-specific parameters before sending to upstream provider"""
        # Define proxy-specific parameters that should not be forwarded
        proxy_params = {"cache"}
        
        # Create a copy and remove proxy-specific parameters
        filtered_data = {}
        for key, value in request_data.items():
            if key not in proxy_params:
                # Handle extra_body specially - filter out proxy cache params
                if key == "extra_body" and isinstance(value, dict):
                    filtered_extra_body = {k: v for k, v in value.items() if k != "cache"}
                    if filtered_extra_body:  # Only include if not empty
                        filtered_data[key] = filtered_extra_body
                else:
                    filtered_data[key] = value
        
        return filtered_data

    async def _make_request(
        self, endpoint: Endpoint, request_data: dict, is_streaming: bool
    ) -> dict:
        """Make request to a specific endpoint"""

        # Pre-emptively consume rate limit capacity
        await self.rate_limit_manager.consume_capacity(endpoint.id)

        # Extract endpoint parameters
        api_key = endpoint.params.get("api_key")
        base_url = endpoint.params.get("base_url", "https://api.openai.com")
        default_query = endpoint.params.get("default_query")

        # Create a copy of request data, filter proxy params, and update the model name
        filtered_data = self._filter_proxy_params(request_data)
        filtered_data["model"] = endpoint.model

        logger.info(
            "making_request",
            endpoint_id=endpoint.id,
            model=request_data.get("model"),
            endpoint_model=endpoint.model,
            streaming=is_streaming,
            endpoint_base_url=base_url,
            request_size=len(str(filtered_data)),
        )

        # Warn about very large requests
        request_size = len(str(filtered_data))
        if request_size > 10_000_000:  # 10MB
            logger.warning(
                "large_chat_completion_request",
                endpoint_id=endpoint.id,
                model=request_data.get("model"),
                size_mb=round(request_size / 1_000_000, 2),
                has_image="image" in str(filtered_data).lower() or "image_url" in str(filtered_data).lower(),
                endpoint_base_url=base_url,
            )

        # Make the request
        response = await self.llm_client.create_chat_completion(
            model=endpoint.model,
            endpoint_url=base_url,
            api_key=api_key,
            request_data=filtered_data,
            default_query=default_query,
            stream=is_streaming,
        )

        # For streaming, the response is already an async generator
        if is_streaming:
            return {
                "status_code": 200,  # Assume success for now
                "data": response,
                "headers": {},
            }

        # Log response details for non-streaming
        if response.get("status_code") != 200:
            logger.debug(
                "chat_completion_request_failed",
                endpoint_id=endpoint.id,
                endpoint_base_url=base_url,
                status_code=response.get("status_code"),
                duration_ms=response.get("duration_ms"),
                error_summary=str(response.get("error", ""))[:200],
            )

        return response
