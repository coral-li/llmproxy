from typing import Callable, Any
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
import json

from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.core.cache_manager import CacheManager
from llmproxy.clients.llm_client import LLMClient
from llmproxy.api.chat_completions import ChatCompletionHandler
from llmproxy.api.responses import ResponseHandler
from llmproxy.core.logger import get_logger

logger = get_logger(__name__)


def create_router(
    get_load_balancer: Callable[[], LoadBalancer],
    cache_manager: Callable[[], CacheManager] = None,
    llm_client: Callable[[], LLMClient] = None,
    config: Callable[[], Any] = None
) -> APIRouter:
    """Create FastAPI router with actual LLM proxy endpoints"""
    
    router = APIRouter()
    
    # Initialize handlers lazily
    chat_handler = None
    response_handler = None
    
    def get_chat_handler():
        nonlocal chat_handler
        if chat_handler is None:
            lb = get_load_balancer()
            if not lb:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not ready"
                )
            chat_handler = ChatCompletionHandler(
                load_balancer=lb,
                cache_manager=cache_manager() if cache_manager else None,
                llm_client=llm_client() if llm_client else None,
                config=config() if config else None
            )
        return chat_handler
    
    def get_response_handler():
        nonlocal response_handler
        if response_handler is None:
            lb = get_load_balancer()
            if not lb:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not ready"
                )
            response_handler = ResponseHandler(
                load_balancer=lb,
                cache_manager=cache_manager() if cache_manager else None,
                llm_client=llm_client() if llm_client else None,
                config=config() if config else None
            )
        return response_handler

    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint"""
        try:
            request_data = await request.json()
            handler = get_chat_handler()
            return await handler.handle_request(request_data)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("chat_completion_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @router.post("/completions")
    async def completions(request: Request):
        """OpenAI-compatible completions endpoint (legacy)"""
        # For now, redirect to chat completions with a simple conversion
        try:
            request_data = await request.json()
            
            # Convert legacy completions to chat format
            prompt = request_data.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]
            
            # Update request data
            request_data["messages"] = messages
            request_data.pop("prompt", None)
            
            handler = get_chat_handler()
            response = await handler.handle_request(request_data)
            
            # Convert response back to completions format if not streaming
            if isinstance(response, dict) and "choices" in response:
                # Convert chat response to completions format
                for choice in response.get("choices", []):
                    if "message" in choice:
                        choice["text"] = choice["message"].get("content", "")
                        choice.pop("message", None)
                response["object"] = "text_completion"
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("completion_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @router.post("/responses")
    async def responses(request: Request):
        """OpenAI Responses API endpoint"""
        try:
            request_data = await request.json()
            handler = get_response_handler()
            return await handler.handle_request(request_data)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("responses_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    return router 