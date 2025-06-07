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


class ResponseHandler:
    """Handles response API requests with load balancing, caching, and retries"""

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
        """Handle incoming response API request"""
        start_time = time.time()

        # Extract model and determine model group
        model = request_data.get("model")
        if not model:
            raise HTTPException(400, "Model is required")

        # Map model to model group
        model_group = model

        # Check if model group exists
        if model_group not in self.load_balancer.get_model_groups():
            raise HTTPException(400, f"Model '{model}' not configured")

        is_streaming = request_data.get("stream", False)

        # Check cache for non-streaming requests
        if self.config.general_settings.cache and not is_streaming:
            cached_response = await self.cache_manager.get(request_data)
            if cached_response:
                data = cached_response.get("data")
                # Add proxy metadata
                data["_proxy_cache_hit"] = True
                data["_proxy_latency_ms"] = int((time.time() - start_time) * 1000)
                return data

        # Execute request with retries
        response = await self._execute_with_failover(model_group, request_data)

        # For streaming responses, return immediately
        if is_streaming and isinstance(response, StreamingResponse):
            return response

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
            # For streaming, check if it's already a StreamingResponse
            if isinstance(response, StreamingResponse):
                return response
            else:
                # If not, something went wrong, return error
                raise HTTPException(500, "Streaming response error")
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
                    # For streaming, return a streaming response
                    streaming_response = StreamingResponse(
                        response["data"],
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                            "X-Proxy-Endpoint-Base-Url": endpoint.params.get("base_url", "https://api.openai.com"),
                        },
                    )
                    return streaming_response

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
                    # Rate limit hit
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
                        endpoint_base_url=endpoint.params.get("base_url"),
                        model_group=model_group,
                        attempt=attempt + 1,
                        max_attempts=self.config.general_settings.num_retries,
                        duration_ms=response.get("duration_ms"),
                        response_headers=response.get("headers", {}),
                        request_model=request_data.get("model"),
                    )

                    self.load_balancer.record_failure(endpoint, error_msg)

                else:
                    # Non-retryable error - still retry with different endpoint
                    # Don't record failure since it's likely a client error, not endpoint fault
                    logger.warning(
                        "non_retryable_error_retrying",
                        endpoint_id=endpoint.id,
                        status_code=response["status_code"],
                        error=error_msg[:200],
                        endpoint_base_url=endpoint.params.get("base_url"),
                        model_group=model_group,
                        attempt=attempt + 1,
                        max_attempts=self.config.general_settings.num_retries,
                        duration_ms=response.get("duration_ms"),
                        response_headers=response.get("headers", {}),
                        request_model=request_data.get("model"),
                    )

                # Store the last response to return if all endpoints fail
                last_response = response

            except Exception as e:
                logger.error("request_exception", endpoint_id=endpoint.id, error=str(e))

                self.load_balancer.record_failure(endpoint, str(e))

                # For streaming, return error in SSE format
                if is_streaming:
                    error_message = str(e)  # Capture error message for the inner function

                    async def error_stream():
                        yield f'data: {{"error": {{"message": "{error_message}", "type": "proxy_error"}}}}\n\n'
                        yield "data: [DONE]\n\n"

                    return {"status_code": 500, "data": error_stream()}

        # All retries exhausted - return the last response we received
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
            "making_response_request",
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
                "large_response_request",
                endpoint_id=endpoint.id,
                model=request_data.get("model"),
                size_mb=round(request_size / 1_000_000, 2),
                has_image="image" in str(filtered_data).lower() or "image_url" in str(filtered_data).lower(),
                endpoint_base_url=base_url,
            )

        # Make the request
        response = await self.llm_client.create_response(
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
                "response_request_failed",
                endpoint_id=endpoint.id,
                endpoint_base_url=base_url,
                status_code=response.get("status_code"),
                duration_ms=response.get("duration_ms"),
                error_summary=str(response.get("error", ""))[:200],
            )

        return response 