from llmproxy.api.base_handler import BaseRequestHandler
from llmproxy.core.logger import get_logger
from llmproxy.models.endpoint import Endpoint

logger = get_logger(__name__)


class EmbeddingHandler(BaseRequestHandler):
    """Handles embedding requests with load balancing, caching, and retries"""

    async def _make_request(
        self, endpoint: Endpoint, request_data: dict, is_streaming: bool = False
    ) -> dict:
        """Make request to a specific endpoint"""
        # Embeddings are never streamed, so is_streaming is ignored.

        # Extract endpoint parameters
        base_url = endpoint.params.get("base_url", "https://api.openai.com")

        # Create a copy of request data, filter proxy params, and update the model name
        filtered_data = self._filter_proxy_params(request_data)
        filtered_data["model"] = endpoint.model

        logger.info(
            "making_embedding_request",
            endpoint_id=endpoint.id,
            model=request_data.get("model"),
            endpoint_model=endpoint.model,
            endpoint_base_url=base_url,
            request_size=len(str(filtered_data)),
        )

        # Make the request using endpoint-aware client API
        response = await self.llm_client.create_embedding(
            endpoint=endpoint,
            request_data=filtered_data,
        )

        # Log response details
        if response.get("status_code") != 200:
            logger.debug(
                "embedding_request_failed",
                endpoint_id=endpoint.id,
                endpoint_base_url=base_url,
                status_code=response.get("status_code"),
                duration_ms=response.get("duration_ms"),
                error_summary=str(response.get("error", ""))[:200],
            )

        return response
