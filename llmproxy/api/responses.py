from llmproxy.api.base_handler import BaseRequestHandler
from llmproxy.core.logger import get_logger
from llmproxy.models.endpoint import Endpoint

logger = get_logger(__name__)


class ResponseHandler(BaseRequestHandler):
    """Handles response generation requests with load balancing, caching, and retries"""

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
                "data": response,
                "headers": {},
            }

        # Log response details for non-streaming
        if isinstance(response, dict):
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
