import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from llmproxy.api.error_handler import is_request_error, is_retryable_error
from llmproxy.clients.llm_client import LLMClient
from llmproxy.config_model import LLMProxyConfig
from llmproxy.core.cache_manager import CacheManager
from llmproxy.core.logger import get_logger
from llmproxy.core.usage_telemetry import (
    ServedBy,
    UsageContext,
    UsageRecorder,
    error_code,
    normalize_usage,
    usage_from_chunks,
)
from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.models.endpoint import Endpoint

logger = get_logger(__name__)

#: Headers on every streamed answer, so nothing between proxy and client
#: buffers it.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _elapsed_ms(start_time: float) -> int:
    return int((time.time() - start_time) * 1000)


def _served_by(endpoint: Endpoint) -> ServedBy:
    return ServedBy(
        endpoint_model=endpoint.model,
        endpoint_id=endpoint.id,
        endpoint_base_url=endpoint.upstream_base_url,
    )


async def _replay(chunks: List[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


def _error_label(response: dict) -> Optional[str]:
    """Return a content-free label for why a response failed, or None.

    Failures the proxy generates itself carry their own label; an upstream
    failure is labelled with the provider's error code, or its status when it
    sends none.
    """
    status_code = response.get("status_code", 500)
    if status_code == 200:
        return None
    return (
        response.get("error_label")
        or error_code(response.get("error"))
        or f"http_{status_code}"
    )


class BaseRequestHandler(ABC):
    """Base class for request handlers with load balancing, caching, and retries"""

    # Identifies the OpenAI-compatible surface this handler serves, so usage
    # records can be split by API without inspecting the request body.
    api_surface: str = "unknown"

    def __init__(
        self,
        load_balancer: LoadBalancer,
        cache_manager: CacheManager,
        llm_client: LLMClient,
        config: LLMProxyConfig,
        usage_recorder: Optional[UsageRecorder] = None,
    ):
        self.load_balancer = load_balancer
        self.cache_manager = cache_manager
        self.llm_client = llm_client
        self.config = config
        self.usage_recorder = usage_recorder or UsageRecorder.disabled()

    async def handle_request(
        self,
        request_data: dict,
        request_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        """Handle incoming request with caching and failover"""
        start_time = time.time()

        model = request_data.get("model")
        if not model:
            raise HTTPException(400, "Model is required")
        model_group = model
        if model_group not in self.load_balancer.get_model_groups():
            raise HTTPException(400, f"Model '{model}' not configured")
        is_streaming = request_data.get("stream", False)

        usage_context = UsageContext.from_request(
            api_surface=self.api_surface,
            model_group=model_group,
            request_data=request_data,
            caller=self.usage_recorder.caller_headers(request_headers),
            start_time=start_time,
        )

        # Check cache; a hit is recorded where it is detected.
        cached_response = await self._check_cache(
            request_data, is_streaming, start_time, model, usage_context
        )
        if cached_response is not None:
            return cached_response

        # Execute request with retries
        response = await self._execute_with_failover(
            model_group, request_data, usage_context
        )

        # For streaming responses, return immediately. The usage record is
        # written by the stream observer once the stream ends.
        if is_streaming and isinstance(response, StreamingResponse):
            return response

        # Ensure we have a dict response for non-streaming operations
        if not isinstance(response, dict):
            raise HTTPException(
                500, "Unexpected response type for non-streaming request"
            )

        # Cache successful non-streaming responses
        await self._maybe_cache_response(request_data, is_streaming, response)

        # Recorded before finalizing, which raises for anything but a 200 and
        # would otherwise drop the record of every failure.
        self._record_completed_response(usage_context, response)

        # Add proxy metadata
        self._add_proxy_metadata(response, start_time)

        # Return appropriate response
        return self._finalize_response(is_streaming, response)

    def _record_cache_hit(
        self, context: UsageContext, cached: Union[Dict[str, Any], List[str]]
    ) -> None:
        """Record a proxy cache hit, which consumes no upstream tokens.

        The usage reported is what the cached body claims; consumers price on
        `cache_hit` so a replayed response is not billed twice, and can use
        these counts to quantify what the cache saved.
        """
        if not self.usage_recorder.enabled:
            return
        usage = (
            normalize_usage(cached)
            if isinstance(cached, dict)
            else usage_from_chunks(cached)
        )
        self.usage_recorder.record(
            context, status_code=200, attempts=0, usage=usage, cache_hit=True
        )

    def _record_completed_response(
        self, usage_context: UsageContext, response: dict
    ) -> None:
        """Record the outcome of a non-streaming (or failed streaming) request."""
        data = response.get("data")
        self.usage_recorder.record(
            usage_context,
            status_code=response["status_code"],
            attempts=response["attempts"],
            served_by=ServedBy.from_response(response),
            usage=normalize_usage(data) if isinstance(data, dict) else None,
            error=_error_label(response),
        )

    async def _check_cache(
        self,
        request_data: dict,
        is_streaming: bool,
        start_time: float,
        model: str,
        usage_context: UsageContext,
    ) -> Optional[Union[Dict[Any, Any], StreamingResponse]]:
        """Serve a cached response if there is one, recording the hit."""
        # Short-circuit if caching is disabled for this request or globally.
        if not self._is_cache_enabled(request_data):
            return None

        if is_streaming:
            cached_chunks = await self.cache_manager.get_streaming(request_data)
            if not cached_chunks:
                return None
            streaming_response = await self._build_cached_streaming_response(
                cached_chunks, request_data, start_time, model
            )
            self._record_cache_hit(usage_context, cached_chunks)
            return streaming_response

        cached_response = await self.cache_manager.get(request_data)
        if not cached_response:
            return None
        cached_response["_proxy_cache_hit"] = True
        cached_response["_proxy_latency_ms"] = _elapsed_ms(start_time)
        await self._on_cached_response(cached_response, request_data, model)
        self._record_cache_hit(usage_context, cached_response)
        return cached_response

    async def _build_cached_streaming_response(
        self,
        cached_chunks: List[str],
        request_data: dict,
        start_time: float,
        model: str,
    ) -> StreamingResponse:
        """Replay a cached stream."""
        lines = await self._cached_stream_lines(cached_chunks, request_data, model)

        async def stream_cached_response() -> AsyncGenerator[bytes, None]:
            async for line in lines:
                yield line.encode("utf-8")

        logger.info(
            "serving_cached_streaming_response",
            model=model,
            num_chunks=len(cached_chunks),
            latency_ms=_elapsed_ms(start_time),
        )
        return StreamingResponse(
            stream_cached_response(),
            media_type="text/event-stream",
            headers={
                **_SSE_HEADERS,
                "X-Proxy-Cache-Hit": "true",
                "X-Proxy-Latency-Ms": str(_elapsed_ms(start_time)),
            },
        )

    async def _cached_stream_lines(
        self, cached_chunks: List[str], request_data: dict, model: str
    ) -> AsyncIterator[str]:
        """The lines a cached stream replays; a subclass may wrap them."""
        return _replay(cached_chunks)

    async def _on_cached_response(
        self, cached_response: dict, request_data: dict, model: str
    ) -> None:
        """Hook run when a non-streaming response is served from cache."""

    async def _maybe_cache_response(
        self, request_data: dict, is_streaming: bool, response: dict
    ) -> None:
        """Cache successful responses if caching is enabled"""
        # Only cache if globally enabled AND not disabled per-request
        if (
            not is_streaming
            and response.get("status_code") == 200
            and self._is_cache_enabled(request_data)
        ):
            await self.cache_manager.set(request_data, response["data"])

    def _add_proxy_metadata(self, response: dict, start_time: float) -> None:
        """Add proxy metadata to response"""
        # A failed streaming attempt carries an error stream here, not a body.
        if isinstance(response.get("data"), dict):
            response["data"]["_proxy_cache_hit"] = False
            response["data"]["_proxy_latency_ms"] = _elapsed_ms(start_time)
            if response.get("endpoint_base_url"):
                response["data"]["_proxy_endpoint_base_url"] = response[
                    "endpoint_base_url"
                ]

    def _finalize_response(
        self, is_streaming: bool, response: dict
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        """Finalize response for return"""
        if is_streaming:
            if isinstance(response, StreamingResponse):
                return response
            if isinstance(response, dict):
                status_code = response.get("status_code", 500)
                detail = response.get("error") or "Streaming request failed"
                raise HTTPException(status_code, detail)
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
        usage_context: UsageContext,
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        """Execute request with automatic failover on errors"""

        # Keep track of endpoints we've already tried for this request
        attempted_endpoints: Set[str] = set()
        endpoints_exhausted = False

        # Calculate max selection attempts
        endpoint_pool = getattr(self.load_balancer, "endpoint_configs", {})
        if isinstance(endpoint_pool, dict):
            total_endpoints = len(endpoint_pool.get(model_group, []))
        else:
            total_endpoints = 0
        max_selection_attempts = max(10, total_endpoints * 2)

        last_response = None

        for attempt in range(self.config.general_settings.num_retries):
            if total_endpoints and len(attempted_endpoints) >= total_endpoints:
                endpoints_exhausted = True
                logger.warning(
                    "all_endpoints_exhausted_before_retry",
                    model_group=model_group,
                    attempted_count=len(attempted_endpoints),
                    total_endpoints=total_endpoints,
                    attempt=attempt + 1,
                )
                break
            # Try to find an endpoint that hasn't been attempted yet
            endpoint, selection_stalled, duplicate_selection_attempts = (
                await self._select_candidate_endpoint(
                    model_group, attempted_endpoints, max_selection_attempts
                )
            )

            if endpoint is None:
                if selection_stalled:
                    endpoints_exhausted = True
                    logger.warning(
                        "load_balancer_selection_exhausted",
                        model_group=model_group,
                        attempted_count=len(attempted_endpoints),
                        max_selection_attempts=max_selection_attempts,
                        duplicate_selection_attempts=duplicate_selection_attempts,
                        attempt=attempt + 1,
                    )
                    break
                return self._no_endpoint_response(
                    model_group, len(attempted_endpoints), last_response
                )

            attempted_endpoints.add(endpoint.id)
            outcome = await self._attempt(
                endpoint, request_data, model_group, attempt, usage_context
            )
            if isinstance(outcome, StreamingResponse) or outcome["status_code"] == 200:
                return outcome
            last_response = outcome
        if endpoints_exhausted or last_response is None:
            return self._all_endpoints_failed_response(
                len(attempted_endpoints), last_response
            )
        return last_response

    async def _attempt(
        self,
        endpoint: Endpoint,
        request_data: dict,
        model_group: str,
        attempt: int,
        usage_context: UsageContext,
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        """Send the request to `endpoint` as attempt number `attempt`, from 0.

        A success comes back ready to serve, and a failure comes back for the
        caller to fail over from. Either way the attempt is stamped with its
        endpoint and with the number of upstream requests made so far.
        """
        is_streaming = request_data.get("stream", False)
        try:
            response = self._stamp_attempt(
                await self._make_request(endpoint, request_data, is_streaming),
                endpoint,
                attempt + 1,
            )
            if response["status_code"] != 200:
                return await self._handle_error_response(
                    endpoint, response, request_data, model_group, attempt, is_streaming
                )
            if is_streaming:
                return await self._build_streaming_response(
                    endpoint, request_data, response, usage_context
                )
            await self.load_balancer.record_success(endpoint)
            return response
        except Exception as e:
            return self._stamp_attempt(
                await self._handle_exception(endpoint, e, is_streaming),
                endpoint,
                attempt + 1,
            )

    @staticmethod
    def _stamp_attempt(response: dict, endpoint: Endpoint, attempts: int) -> dict:
        """Record on a response which endpoint it came from and the attempt count.

        Failures are stamped as well as successes: a failed call that names no
        endpoint is invisible to anything measuring per-endpoint reliability.
        """
        response.update(asdict(_served_by(endpoint)), attempts=attempts)
        return response

    async def _select_candidate_endpoint(
        self,
        model_group: str,
        attempted_endpoints: Set[str],
        max_selection_attempts: int,
    ) -> Tuple[Optional[Endpoint], bool, int]:
        duplicate_attempts = 0
        for _ in range(max_selection_attempts):
            candidate = await self.load_balancer.select_endpoint(
                model_group, exclude_ids=attempted_endpoints
            )
            if not candidate:
                return None, False, duplicate_attempts
            if candidate.id in attempted_endpoints:
                duplicate_attempts += 1
                continue
            return candidate, False, duplicate_attempts
        return None, True, duplicate_attempts

    def _no_endpoint_response(
        self, model_group: str, attempts: int, last_attempt: Optional[dict]
    ) -> dict:
        """Response when no endpoints are available"""
        logger.error("no_endpoint_available", model_group=model_group)
        return self._out_of_endpoints(
            "No available endpoints", "no_available_endpoints", attempts, last_attempt
        )

    async def _build_streaming_response(
        self,
        endpoint: Endpoint,
        request_data: dict,
        response: dict,
        usage_context: UsageContext,
    ) -> StreamingResponse:
        """Build streaming response with caching support"""
        # Only cache if globally enabled AND not disabled per-request
        stream: AsyncIterator[Any]
        if self._is_cache_enabled(request_data):
            cache_writer = await self.cache_manager.create_streaming_cache_writer(
                request_data
            )

            async def cached_stream() -> AsyncGenerator[bytes, None]:
                async for chunk in cache_writer.intercept_stream(response["data"]):
                    yield chunk.encode("utf-8")

            stream = cached_stream()
        else:
            stream = response["data"]

        stream = self.usage_recorder.observe_stream(
            stream,
            usage_context,
            attempts=response["attempts"],
            served_by=_served_by(endpoint),
        )

        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                **_SSE_HEADERS,
                "X-Proxy-Endpoint-Base-Url": endpoint.upstream_base_url,
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
        """Handle error response from endpoint"""
        error_msg = response.get("error", "Unknown error")
        status_code = response["status_code"]

        # Special handling for rate limits
        if status_code == 429:
            logger.warning("rate_limit_hit", endpoint_id=endpoint.id)
            await self.load_balancer.record_failure(endpoint, "Rate limit exceeded")
        elif is_retryable_error(status_code):
            await self.load_balancer.record_failure(endpoint, error_msg)
            logger.warning(
                "retryable_error",
                endpoint_id=endpoint.id,
                status_code=status_code,
                error=error_msg[:200],
                endpoint_base_url=endpoint.params.get("base_url"),
                model_group=model_group,
                attempt=attempt + 1,
                max_attempts=self.config.general_settings.num_retries,
                duration_ms=response.get("duration_ms"),
                response_headers=response.get("headers", {}),
                request_model=request_data.get("model"),
            )
        else:
            logger.warning(
                "non_retryable_error_retrying",
                endpoint_id=endpoint.id,
                status_code=status_code,
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
        """Handle exception during request"""
        logger.error("request_exception", endpoint_id=endpoint.id, error=str(e))
        await self.load_balancer.record_failure(endpoint, str(e))

        if is_streaming:
            import json

            error_message = str(e)

            async def error_stream() -> AsyncGenerator[bytes, None]:
                # Emit a JSON-encoded error object followed by the sentinel [DONE].
                payload = json.dumps(
                    {
                        "error": {
                            "message": error_message,
                            "type": "proxy_error",
                        }
                    }
                )
                # First SSE event with the error payload
                yield f"data: {payload}\n\n".encode("utf-8")
                # Final sentinel indicating stream completion
                yield "data: [DONE]\n\n".encode("utf-8")

            return {
                "status_code": 500,
                "data": error_stream(),
                "error_label": type(e).__name__,
            }
        return {
            "status_code": 500,
            "headers": {},
            "data": None,
            "error": str(e),
            "error_label": type(e).__name__,
        }

    def _all_endpoints_failed_response(
        self, attempts: int, last_attempt: Optional[dict]
    ) -> dict:
        """Response when all endpoints have failed"""
        return self._out_of_endpoints(
            "All endpoints failed after retries",
            "all_endpoints_failed",
            attempts,
            last_attempt,
        )

    @staticmethod
    def _out_of_endpoints(
        error: str, label: str, attempts: int, last_attempt: Optional[dict]
    ) -> dict:
        """The answer once no endpoint is left to try.

        A request the last upstream refused would be refused by every endpoint,
        so the caller gets that refusal rather than a 503 that reads as an
        outage and invites a retry. Any other failure is a 503 that names the
        endpoint tried last, and why it failed: with a single endpoint per
        model group every upstream failure ends here, so without that the
        endpoint that failed would be recorded nowhere.
        """
        if last_attempt is not None and is_request_error(last_attempt["status_code"]):
            return last_attempt
        failure = {
            "status_code": 503,
            "headers": {},
            "data": None,
            "error": error,
            "error_label": label,
            "attempts": attempts,
        }
        if last_attempt is not None:
            failure.update(
                asdict(ServedBy.from_response(last_attempt)),
                error_label=_error_label(last_attempt),
            )
        return failure

    def _filter_proxy_params(self, request_data: dict) -> dict:
        """Filter out proxy-specific parameters from request data"""
        # Define proxy-specific parameters that should not be forwarded to upstream APIs
        proxy_params = {"cache"}

        # Create a copy and remove proxy-specific parameters
        filtered_data = {}
        for key, value in request_data.items():
            if key not in proxy_params:
                # Handle extra_body specially - filter out proxy cache params but preserve other content
                if key == "extra_body" and isinstance(value, dict):
                    filtered_extra_body = {
                        k: v for k, v in value.items() if k != "cache"
                    }
                    if filtered_extra_body:  # Only include if not empty
                        filtered_data[key] = filtered_extra_body
                else:
                    filtered_data[key] = value

        return filtered_data

    @abstractmethod
    async def _make_request(
        self, endpoint: Endpoint, request_data: dict, is_streaming: bool
    ) -> dict:
        """Make request to a specific endpoint"""
        pass

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _is_cache_enabled(self, request_data: dict) -> bool:
        """Return True if global and per-request caching are both enabled."""
        return self.config.general_settings.cache and self.cache_manager._should_cache(
            request_data
        )
