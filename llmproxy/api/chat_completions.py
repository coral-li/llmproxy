import time
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from llmproxy.api.error_handler import is_retryable_error
from llmproxy.clients.llm_client import LLMClient
from llmproxy.config_model import LLMProxyConfig
from llmproxy.core.cache_manager import CacheManager
from llmproxy.core.logger import get_logger
from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.models.endpoint import Endpoint

# Imports are handled properly through package structure


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

    async def handle_request(
        self, request_data: dict
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        """Handle incoming chat completion request"""
        start_time = time.time()

        model = request_data.get("model")
        if not model:
            raise HTTPException(400, "Model is required")
        model_group = model
        if model_group not in self.load_balancer.get_model_groups():
            raise HTTPException(400, f"Model '{model}' not configured")
        is_streaming = request_data.get("stream", False)

        # Check cache
        cached_response = await self._check_cache(
            request_data, is_streaming, start_time, model
        )
        if cached_response is not None:
            return cached_response

        # Execute request with retries
        response = await self._execute_with_failover(model_group, request_data)

        # For streaming responses, return immediately
        if is_streaming and isinstance(response, StreamingResponse):
            return response

        # Ensure we have a dict response for non-streaming operations
        if not isinstance(response, dict):
            raise HTTPException(
                500, "Unexpected response type for non-streaming request"
            )

        # Cache successful non-streaming responses
        await self._maybe_cache_response(request_data, is_streaming, response)

        # Add proxy metadata
        self._add_proxy_metadata(response, start_time)

        # Return appropriate response
        return self._finalize_response(is_streaming, response)

    async def _check_cache(
        self, request_data: dict, is_streaming: bool, start_time: float, model: str
    ) -> Optional[Union[Dict[Any, Any], StreamingResponse]]:
        if self.config.general_settings.cache and is_streaming:
            cached_chunks = await self.cache_manager.get_streaming(request_data)
            if cached_chunks:

                async def stream_cached_response() -> AsyncGenerator[bytes, None]:
                    for chunk in cached_chunks:
                        # cached_chunks from get_streaming always contains strings
                        yield chunk.encode("utf-8")

                logger.info(
                    "serving_cached_streaming_response",
                    model=model,
                    num_chunks=len(cached_chunks),
                    latency_ms=int((time.time() - start_time) * 1000),
                )
                return StreamingResponse(
                    stream_cached_response(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "X-Proxy-Cache-Hit": "true",
                        "X-Proxy-Latency-Ms": str(
                            int((time.time() - start_time) * 1000)
                        ),
                    },
                )
        if self.config.general_settings.cache and not is_streaming:
            cached_response = await self.cache_manager.get(request_data)
            if cached_response:
                cached_response["_proxy_cache_hit"] = True
                cached_response["_proxy_latency_ms"] = int(
                    (time.time() - start_time) * 1000
                )
                return cached_response
        return None

    async def _maybe_cache_response(
        self, request_data: dict, is_streaming: bool, response: dict
    ) -> None:
        if (
            self.config.general_settings.cache
            and not is_streaming
            and response.get("status_code") == 200
        ):
            await self.cache_manager.set(request_data, response["data"])

    def _add_proxy_metadata(self, response: dict, start_time: float) -> None:
        if response.get("data"):
            response["data"]["_proxy_cache_hit"] = False
            response["data"]["_proxy_latency_ms"] = int(
                (time.time() - start_time) * 1000
            )
            if response.get("endpoint_base_url"):
                response["data"]["_proxy_endpoint_base_url"] = response[
                    "endpoint_base_url"
                ]

    def _finalize_response(
        self, is_streaming: bool, response: dict
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        if is_streaming:
            if isinstance(response, StreamingResponse):
                return response
            else:
                raise HTTPException(500, "Streaming response error")
        else:
            if response["status_code"] == 200:
                data = response["data"]
                if isinstance(data, dict):
                    return data
                else:
                    raise HTTPException(500, "Invalid response data type")
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
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        """Execute request with automatic failover on errors"""

        attempted_endpoints = attempted_endpoints or set()
        is_streaming = request_data.get("stream", False)
        last_response = None

        for attempt in range(self.config.general_settings.num_retries):
            endpoint = await self.load_balancer.select_endpoint(model_group)
            if not endpoint:
                return self._no_endpoint_response(model_group)
            if endpoint.id in attempted_endpoints:
                continue
            attempted_endpoints.add(endpoint.id)
            try:
                response = await self._make_request(
                    endpoint, request_data, is_streaming
                )
                if is_streaming and response.get("status_code") == 200:
                    return await self._build_streaming_response(
                        endpoint, request_data, response
                    )
                if response["status_code"] == 200:
                    await self.load_balancer.record_success(endpoint)
                    response["endpoint_base_url"] = endpoint.params.get(
                        "base_url", "https://api.openai.com"
                    )
                    return response
                last_response = await self._handle_error_response(
                    endpoint, response, request_data, model_group, attempt, is_streaming
                )
            except Exception as e:
                last_response = await self._handle_exception(endpoint, e, is_streaming)
        return last_response or self._all_endpoints_failed_response()

    def _no_endpoint_response(self, model_group: str) -> dict:
        logger.error("no_endpoint_available", model_group=model_group)
        return {
            "status_code": 503,
            "headers": {},
            "data": None,
            "error": "No available endpoints",
        }

    async def _build_streaming_response(
        self, endpoint: Endpoint, request_data: dict, response: dict
    ) -> StreamingResponse:
        should_cache = self.cache_manager._should_cache(request_data)
        if self.config.general_settings.cache and should_cache:
            cache_writer = await self.cache_manager.create_streaming_cache_writer(
                request_data
            )

            async def cached_stream() -> AsyncGenerator[bytes, None]:
                async for chunk in cache_writer.intercept_stream(response["data"]):
                    # intercept_stream always yields strings
                    yield chunk.encode("utf-8")

            stream = cached_stream()
        else:
            stream = response["data"]
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Proxy-Endpoint-Base-Url": endpoint.params.get(
                    "base_url", "https://api.openai.com"
                ),
                "X-Proxy-Cache-Hit": "false",
            },
        )

    async def _handle_error_response(
        self,
        endpoint: Endpoint,
        response: dict,
        request_data: dict,
        model_group: str,
        attempt: int,
        is_streaming: bool,
    ) -> dict:
        error_msg = response.get("error", "Unknown error")
        if is_retryable_error(response["status_code"]):
            await self.load_balancer.record_failure(endpoint, error_msg)
            logger.warning(
                "retryable_error",
                endpoint_id=endpoint.id,
                status_code=response["status_code"],
                error=error_msg[:200],
                attempt=attempt + 1,
                max_attempts=self.config.general_settings.num_retries,
            )
        else:
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
        return response

    async def _handle_exception(
        self, endpoint: Endpoint, e: Exception, is_streaming: bool
    ) -> dict:
        logger.error("request_exception", endpoint_id=endpoint.id, error=str(e))
        await self.load_balancer.record_failure(endpoint, str(e))
        if is_streaming:
            error_message = str(e)

            async def error_stream() -> AsyncGenerator[bytes, None]:
                yield f'data: {{"error": {{"message": "{error_message}", "type": "proxy_error"}}}}\n\n'.encode(
                    "utf-8"
                )
                yield "data: [DONE]\n\n".encode("utf-8")

            return {"status_code": 500, "data": error_stream()}
        return {
            "status_code": 500,
            "headers": {},
            "data": None,
            "error": str(e),
        }

    def _all_endpoints_failed_response(self) -> dict:
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
        api_key = endpoint.params.get("api_key", "")
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
                has_image="image" in str(filtered_data).lower()
                or "image_url" in str(filtered_data).lower(),
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
        if isinstance(response, dict):
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
        else:
            # This shouldn't happen for non-streaming, but handle it gracefully
            return {
                "status_code": 500,
                "headers": {},
                "data": None,
                "error": "Unexpected response type for non-streaming request",
            }
