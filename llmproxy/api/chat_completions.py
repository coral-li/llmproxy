from typing import Dict, Any, Optional
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import time
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from managers.load_balancer import LoadBalancer
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
        cache_manager: CacheManager,
        llm_client: LLMClient,
        config: LLMProxyConfig,
    ):
        self.load_balancer = load_balancer
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

        # Check cache for streaming requests
        if self.config.general_settings.cache and is_streaming:
            cached_chunks = await self.cache_manager.get_streaming(request_data)
            if cached_chunks:
                # Create a streaming response from cached chunks
                async def stream_cached_response():
                    for chunk in cached_chunks:
                        yield chunk
                
                logger.info(
                    "serving_cached_streaming_response",
                    model=model,
                    num_chunks=len(cached_chunks),
                    latency_ms=int((time.time() - start_time) * 1000)
                )
                
                return StreamingResponse(
                    stream_cached_response(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "X-Proxy-Cache-Hit": "true",
                        "X-Proxy-Latency-Ms": str(int((time.time() - start_time) * 1000)),
                    },
                )

        # Check cache for non-streaming requests
        if self.config.general_settings.cache and not is_streaming:
            cached_response = await self.cache_manager.get(request_data)
            if cached_response:
                # Add proxy metadata
                cached_response["_proxy_cache_hit"] = True
                cached_response["_proxy_latency_ms"] = int(
                    (time.time() - start_time) * 1000
                )
                return cached_response

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
            await self.cache_manager.set(request_data, response["data"])

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
                    # Check if caching is enabled for this request
                    should_cache = self.cache_manager._should_cache(request_data, ignore_streaming=True)
                    
                    if self.config.general_settings.cache and should_cache:
                        # Create a caching interceptor
                        cache_writer = await self.cache_manager.create_streaming_cache_writer(request_data)
                        
                        # Wrap the stream with caching
                        async def cached_stream():
                            async for chunk in cache_writer.intercept_stream(response["data"]):
                                yield chunk
                        
                        # Return streaming response with caching
                        streaming_response = StreamingResponse(
                            cached_stream(),
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                                "X-Proxy-Endpoint-Base-Url": endpoint.params.get("base_url", "https://api.openai.com"),
                                "X-Proxy-Cache-Hit": "false",
                            },
                        )
                    else:
                        # Return streaming response without caching
                        streaming_response = StreamingResponse(
                            response["data"],
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                                "X-Proxy-Endpoint-Base-Url": endpoint.params.get("base_url", "https://api.openai.com"),
                                "X-Proxy-Cache-Hit": "false",
                            },
                        )
                    
                    return streaming_response

                # Check response status
                if response["status_code"] == 200:
                    # Success!
                    self.load_balancer.record_success(endpoint)

                    # Add endpoint information to response
                    response["endpoint_base_url"] = endpoint.params.get("base_url", "https://api.openai.com")

                    return response

                # Handle errors - retry with different endpoint regardless of error type
                error_msg = response.get("error", "Unknown error")

                # Check if error is retryable
                if is_retryable_error(response["status_code"]):
                    # Record failure and continue to next endpoint
                    self.load_balancer.record_failure(endpoint, error_msg)

                    logger.warning(
                        "retryable_error",
                        endpoint_id=endpoint.id,
                        status_code=response["status_code"],
                        error=error_msg[:200],
                        attempt=attempt + 1,
                        max_attempts=self.config.general_settings.num_retries,
                    )

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
        """Filter out proxy-specific parameters from request data"""
        # Create a copy
        filtered = request_data.copy()

        # Remove proxy-specific fields
        proxy_fields = ["cache", "extra_body"]
        for field in proxy_fields:
            filtered.pop(field, None)

        return filtered

    async def _make_request(
        self, endpoint: Endpoint, request_data: dict, is_streaming: bool
    ) -> dict:
        """Make request to a specific endpoint"""

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
                "large_request",
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
