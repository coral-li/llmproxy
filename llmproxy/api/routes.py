from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, HTTPException, Request, status

from llmproxy.api.chat_completions import ChatCompletionHandler
from llmproxy.api.embeddings import EmbeddingHandler
from llmproxy.api.responses import ResponseHandler
from llmproxy.clients.llm_client import LLMClient
from llmproxy.core.cache_manager import CacheManager
from llmproxy.core.logger import get_logger
from llmproxy.managers.load_balancer import LoadBalancer

logger = get_logger(__name__)


def create_router(
    get_load_balancer: Callable[[], LoadBalancer],
    cache_manager: Optional[Callable[[], CacheManager]] = None,
    llm_client: Optional[Callable[[], LLMClient]] = None,
    config: Optional[Callable[[], Any]] = None,
) -> APIRouter:
    """Create FastAPI router with all LLM proxy endpoints"""

    router = APIRouter()

    chat_handler = None
    response_handler = None
    embedding_handler = None

    def get_chat_handler() -> ChatCompletionHandler:
        nonlocal chat_handler
        if chat_handler is None:
            chat_handler = _create_chat_handler(
                get_load_balancer, cache_manager, llm_client, config
            )
        return chat_handler

    def get_response_handler() -> ResponseHandler:
        nonlocal response_handler
        if response_handler is None:
            response_handler = _create_response_handler(
                get_load_balancer, cache_manager, llm_client, config
            )
        return response_handler

    def get_embedding_handler() -> EmbeddingHandler:
        nonlocal embedding_handler
        if embedding_handler is None:
            embedding_handler = _create_embedding_handler(
                get_load_balancer, cache_manager, llm_client, config
            )
        return embedding_handler

    @router.post("/chat/completions")
    async def chat_completions(request: Request) -> Any:
        return await _handle_endpoint(request, get_chat_handler, _process_chat_request)

    @router.post("/responses")
    async def responses(request: Request) -> Any:
        return await _handle_endpoint(
            request, get_response_handler, _process_response_request
        )

    @router.post("/embeddings")
    async def embeddings(request: Request) -> Any:
        print("embeddings endpoint called")
        return await _handle_endpoint(
            request, get_embedding_handler, _process_embedding_request
        )

    return router


def _create_chat_handler(
    get_load_balancer: Callable[[], LoadBalancer],
    cache_manager: Optional[Callable[[], CacheManager]],
    llm_client: Optional[Callable[[], LLMClient]],
    config: Optional[Callable[[], Any]],
) -> ChatCompletionHandler:
    lb = get_load_balancer()
    if not lb:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    # Get the dependencies, ensuring they're not None
    cm = cache_manager() if cache_manager is not None else None
    lc = llm_client() if llm_client is not None else None
    cfg = config() if config is not None else None

    # Validate required dependencies
    if cm is None or lc is None or cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required dependencies not available",
        )

    return ChatCompletionHandler(
        load_balancer=lb,
        cache_manager=cm,
        llm_client=lc,
        config=cfg,
    )


def _create_response_handler(
    get_load_balancer: Callable[[], LoadBalancer],
    cache_manager: Optional[Callable[[], CacheManager]],
    llm_client: Optional[Callable[[], LLMClient]],
    config: Optional[Callable[[], Any]],
) -> ResponseHandler:
    lb = get_load_balancer()
    if not lb:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    # Get the dependencies, ensuring they're not None
    cm = cache_manager() if cache_manager is not None else None
    lc = llm_client() if llm_client is not None else None
    cfg = config() if config is not None else None

    # Validate required dependencies
    if cm is None or lc is None or cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required dependencies not available",
        )

    return ResponseHandler(
        load_balancer=lb,
        cache_manager=cm,
        llm_client=lc,
        config=cfg,
    )


def _create_embedding_handler(
    get_load_balancer: Callable[[], LoadBalancer],
    cache_manager: Optional[Callable[[], CacheManager]],
    llm_client: Optional[Callable[[], LLMClient]],
    config: Optional[Callable[[], Any]],
) -> EmbeddingHandler:
    lb = get_load_balancer()
    if not lb:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    # Get the dependencies, ensuring they're not None
    cm = cache_manager() if cache_manager is not None else None
    lc = llm_client() if llm_client is not None else None
    cfg = config() if config is not None else None

    # Validate required dependencies
    if cm is None or lc is None or cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required dependencies not available",
        )

    return EmbeddingHandler(
        load_balancer=lb,
        cache_manager=cm,
        llm_client=lc,
        config=cfg,
    )


async def _process_chat_request(
    handler: ChatCompletionHandler, request_data: dict
) -> Any:
    return await handler.handle_request(request_data)


async def _process_response_request(
    handler: ResponseHandler, request_data: dict
) -> Any:
    return await handler.handle_request(request_data)


async def _process_embedding_request(
    handler: EmbeddingHandler, request_data: dict
) -> Any:
    return await handler.handle_request(request_data)


async def _handle_endpoint(
    request: Request,
    get_handler: Callable[[], Any],
    process_func: Callable[[Any, dict], Awaitable[Any]],
) -> Any:
    try:
        request_data = await request.json()
        handler = get_handler()
        return await process_func(handler, request_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{process_func.__name__}_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
