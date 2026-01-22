import json
from typing import Any, AsyncIterator, Dict, Optional, Union

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from llmproxy.api.base_handler import BaseRequestHandler
from llmproxy.clients.llm_client import LLMClient
from llmproxy.config_model import LLMProxyConfig
from llmproxy.core.cache_manager import CacheManager
from llmproxy.core.logger import get_logger
from llmproxy.core.response_affinity import ResponseAffinityManager
from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.models.endpoint import Endpoint

logger = get_logger(__name__)


class ResponseHandler(BaseRequestHandler):
    """Handles response generation requests with load balancing, caching, and retries"""

    def __init__(
        self,
        load_balancer: LoadBalancer,
        cache_manager: CacheManager,
        llm_client: LLMClient,
        config: LLMProxyConfig,
        response_affinity_manager: ResponseAffinityManager,
    ) -> None:
        super().__init__(
            load_balancer=load_balancer,
            cache_manager=cache_manager,
            llm_client=llm_client,
            config=config,
        )
        self.response_affinity_manager = response_affinity_manager

    async def handle_request(
        self, request_data: dict
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        self._validate_stateful_request(request_data)
        return await super().handle_request(request_data)

    async def _make_request(
        self, endpoint: Endpoint, request_data: dict, is_streaming: bool
    ) -> dict:
        """Make request to a specific endpoint"""

        # Extract endpoint parameters
        base_url = endpoint.params.get("base_url", "https://api.openai.com")

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
                has_image="image" in str(filtered_data).lower()
                or "image_url" in str(filtered_data).lower(),
                endpoint_base_url=base_url,
            )

        # Make the request using endpoint-aware client API
        response = await self.llm_client.create_response(
            endpoint=endpoint,
            request_data=filtered_data,
            stream=is_streaming,
        )

        # For streaming, check if we got an error dict or a generator
        if is_streaming:
            # If it's a dict, it's an error response
            if isinstance(response, dict):
                return response
            # Otherwise it's a successful streaming response
            return {
                "status_code": 200,
                "data": self._wrap_stream_for_affinity(response, endpoint.id),
                "headers": {},
            }

        # Log response details for non-streaming
        if isinstance(response, dict):
            if response.get("status_code") == 200:
                await self._record_affinity_from_response(
                    response.get("data"), endpoint
                )
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
        else:
            # This shouldn't happen for non-streaming, but handle it gracefully
            return {
                "status_code": 500,
                "headers": {},
                "data": None,
                "error": "Unexpected response type for non-streaming request",
            }

    async def _execute_with_failover(
        self, model_group: str, request_data: dict
    ) -> Union[Dict[Any, Any], StreamingResponse]:
        previous_response_id = self._get_previous_response_id(request_data)
        if not previous_response_id:
            return await super()._execute_with_failover(model_group, request_data)

        endpoint = await self._get_pinned_endpoint(model_group, previous_response_id)
        is_streaming = request_data.get("stream", False)
        last_response = None

        for attempt in range(self.config.general_settings.num_retries):
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
                    endpoint,
                    response,
                    request_data,
                    model_group,
                    attempt,
                    is_streaming,
                )
            except Exception as exc:
                last_response = await self._handle_exception(
                    endpoint, exc, is_streaming
                )

        return last_response or self._all_endpoints_failed_response()

    def _validate_stateful_request(self, request_data: dict) -> None:
        previous_response_id = self._get_previous_response_id(request_data)
        if not previous_response_id and self._has_encrypted_reasoning(request_data):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Encrypted reasoning content requires previous_response_id "
                    "for affinity routing"
                ),
            )

    def _get_previous_response_id(self, request_data: dict) -> Optional[str]:
        previous_response_id = request_data.get("previous_response_id")
        if isinstance(previous_response_id, str) and previous_response_id.strip():
            return previous_response_id.strip()
        return None

    async def _get_pinned_endpoint(
        self, model_group: str, previous_response_id: str
    ) -> Endpoint:
        endpoint_id = await self.response_affinity_manager.get_endpoint_id(
            previous_response_id
        )
        if not endpoint_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Unknown previous_response_id; affinity mapping expired or missing"
                ),
            )

        for endpoint in self.load_balancer.endpoint_configs.get(model_group, []):
            if endpoint.id == endpoint_id:
                return endpoint

        raise HTTPException(
            status_code=409,
            detail="previous_response_id mapped to an unavailable endpoint",
        )

    def _has_encrypted_reasoning(self, request_data: dict) -> bool:
        input_data = request_data.get("input")
        if input_data is None:
            return False
        return self._contains_encrypted_reasoning(input_data)

    def _contains_encrypted_reasoning(self, value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("type") == "reasoning" and isinstance(
                value.get("encrypted_content"), str
            ):
                return True
            for nested in value.values():
                if self._contains_encrypted_reasoning(nested):
                    return True
            return False
        if isinstance(value, list):
            return any(self._contains_encrypted_reasoning(item) for item in value)
        return False

    async def _record_affinity_from_response(
        self, response_data: Optional[dict], endpoint: Endpoint
    ) -> None:
        response_id = self._extract_response_id(response_data)
        if response_id:
            await self.response_affinity_manager.set_endpoint_id(
                response_id, endpoint.id
            )

    def _extract_response_id(self, response_data: Optional[dict]) -> Optional[str]:
        if not isinstance(response_data, dict):
            return None
        response_id = response_data.get("id")
        if isinstance(response_id, str) and response_id:
            return response_id
        return None

    async def _wrap_stream_for_affinity(
        self, stream: AsyncIterator[str], endpoint_id: str
    ) -> AsyncIterator[str]:
        response_id: Optional[str] = None

        async for line in stream:
            if response_id is None:
                response_id = self._extract_response_id_from_stream_line(line)
                if response_id:
                    await self.response_affinity_manager.set_endpoint_id(
                        response_id, endpoint_id
                    )
            yield line

    def _extract_response_id_from_stream_line(self, line: str) -> Optional[str]:
        stripped = line.strip()
        if not stripped.startswith("data: "):
            return None
        data = stripped[6:].strip()
        if not data or data == "[DONE]":
            return None
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        response_id = payload.get("response_id")
        if isinstance(response_id, str) and response_id:
            return response_id
        response = payload.get("response")
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                return response_id
        return None
