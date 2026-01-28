import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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
        endpoint = await self._resolve_affinity_endpoint(model_group, request_data)
        if endpoint is None:
            return await super()._execute_with_failover(model_group, request_data)

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

    async def _resolve_affinity_endpoint(
        self, model_group: str, request_data: dict
    ) -> Optional[Endpoint]:
        previous_response_id = self._get_previous_response_id(request_data)
        if previous_response_id:
            # Intentionally treat previous_response_id as authoritative for routing; any
            # encrypted reasoning inputs are assumed to belong to that response and are
            # not revalidated here to avoid rejecting legitimate stateful follow-ups.
            return await self._get_pinned_endpoint(model_group, previous_response_id)

        encrypted_contents = self._extract_encrypted_reasoning_inputs(request_data)
        if not encrypted_contents:
            return None

        endpoint_id = await self._resolve_encrypted_affinity_endpoint_id(
            encrypted_contents
        )

        for endpoint in self.load_balancer.endpoint_configs.get(model_group, []):
            if endpoint.id == endpoint_id:
                return endpoint

        raise HTTPException(
            status_code=409,
            detail="Encrypted reasoning items mapped to an unavailable endpoint",
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

    def _extract_encrypted_reasoning_inputs(self, request_data: dict) -> List[str]:
        input_data = request_data.get("input")
        return self._collect_encrypted_reasoning(input_data)

    def _collect_encrypted_reasoning(self, value: Any) -> List[str]:
        encrypted_items: List[str] = []

        if isinstance(value, dict):
            if value.get("type") == "reasoning" and isinstance(
                value.get("encrypted_content"), str
            ):
                encrypted_items.append(value["encrypted_content"])
            for nested in value.values():
                encrypted_items.extend(self._collect_encrypted_reasoning(nested))
        elif isinstance(value, list):
            for item in value:
                encrypted_items.extend(self._collect_encrypted_reasoning(item))
        return encrypted_items

    async def _record_affinity_from_response(
        self, response_data: Optional[dict], endpoint: Endpoint
    ) -> None:
        response_id = self._extract_response_id(response_data)
        if response_id:
            await self.response_affinity_manager.set_endpoint_id(
                response_id, endpoint.id
            )
        for encrypted_content in self._extract_encrypted_reasoning_outputs(
            response_data
        ):
            await self.response_affinity_manager.set_endpoint_id_for_encrypted(
                encrypted_content, endpoint.id
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
        current_event: Optional[str] = None

        async for line in stream:
            if line.startswith("event: "):
                current_event = line[7:].strip()

            line = self._ensure_created_at_in_stream_line(line, current_event)

            if response_id is None:
                response_id = self._extract_response_id_from_stream_line(line)
                if response_id:
                    await self.response_affinity_manager.set_endpoint_id(
                        response_id, endpoint_id
                    )
            for encrypted_content in self._extract_encrypted_reasoning_from_stream_line(
                line
            ):
                await self.response_affinity_manager.set_endpoint_id_for_encrypted(
                    encrypted_content, endpoint_id
                )

            if line.strip() == "":
                current_event = None
            yield line

    def _ensure_created_at_in_stream_line(
        self, line: str, current_event: Optional[str]
    ) -> str:
        if not line.startswith("data: "):
            return line

        data = line[6:].strip()
        if not data or data == "[DONE]":
            return line

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return line

        if not isinstance(payload, dict):
            return line

        event_type = payload.get("type")
        if current_event != "response.created" and event_type != "response.created":
            return line

        response = payload.get("response")
        if not isinstance(response, dict):
            return line

        created_at = response.get("created_at")
        if created_at is not None:
            return line

        created_at_value = self._coerce_timestamp(response.get("created"))
        if created_at_value is None:
            created_at_value = int(time.time())

        response["created_at"] = created_at_value
        payload["response"] = response

        suffix = "\n" if line.endswith("\n") else ""
        return f"data: {json.dumps(payload)}{suffix}"

    def _coerce_timestamp(self, value: Any) -> Optional[int]:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None
        return None

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

    async def _resolve_encrypted_affinity_endpoint_id(
        self, encrypted_contents: List[str]
    ) -> str:
        endpoint_ids = set()
        for encrypted_content in encrypted_contents:
            endpoint_id = (
                await self.response_affinity_manager.get_endpoint_id_for_encrypted(
                    encrypted_content
                )
            )
            if not endpoint_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Unknown encrypted reasoning content; affinity mapping expired "
                        "or missing"
                    ),
                )
            endpoint_ids.add(endpoint_id)

        if len(endpoint_ids) > 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Encrypted reasoning items map to different endpoints; "
                    "cannot safely route request"
                ),
            )

        return endpoint_ids.pop()

    def _extract_encrypted_reasoning_outputs(
        self, response_data: Optional[dict]
    ) -> List[str]:
        if not isinstance(response_data, dict):
            return []
        outputs = response_data.get("output") or response_data.get("outputs") or []
        return self._extract_encrypted_reasoning_from_items(outputs)

    def _extract_encrypted_reasoning_from_items(self, items: Any) -> List[str]:
        encrypted_items: List[str] = []
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return encrypted_items

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "reasoning" and isinstance(
                item.get("encrypted_content"), str
            ):
                encrypted_items.append(item["encrypted_content"])
        return encrypted_items

    def _extract_encrypted_reasoning_from_stream_line(self, line: str) -> List[str]:
        stripped = line.strip()
        if not stripped.startswith("data: "):
            return []
        data = stripped[6:].strip()
        if not data or data == "[DONE]":
            return []
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []

        encrypted_items: List[str] = []

        item = payload.get("item")
        if isinstance(item, dict):
            encrypted_items.extend(self._extract_encrypted_reasoning_from_items(item))

        response = payload.get("response")
        if isinstance(response, dict):
            outputs = response.get("output") or response.get("outputs") or []
            encrypted_items.extend(
                self._extract_encrypted_reasoning_from_items(outputs)
            )

        return encrypted_items
