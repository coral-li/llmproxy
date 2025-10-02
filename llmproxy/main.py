import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llmproxy.api.routes import create_router
from llmproxy.clients.llm_client import LLMClient
from llmproxy.config.config_loader import load_config_async
from llmproxy.core.cache_manager import CacheManager
from llmproxy.core.exceptions import LLMProxyError
from llmproxy.core.logger import get_logger, setup_logging
from llmproxy.core.redis_manager import RedisManager
from llmproxy.managers.endpoint_state_manager import EndpointStateManager
from llmproxy.managers.load_balancer import LoadBalancer

logger = get_logger(__name__)

# Global state
load_balancer: Optional[LoadBalancer] = None
redis_manager: Optional[RedisManager] = None
cache_manager: Optional[CacheManager] = None
llm_client: Optional[LLMClient] = None
endpoint_state_manager: Optional[EndpointStateManager] = None
config: Optional[Any] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: str
    version: str
    redis_connected: bool
    state_sharing_enabled: bool
    endpoints_configured: int
    model_groups: list


async def init_redis(config: Any) -> Tuple[RedisManager, EndpointStateManager]:
    """Initialize Redis connection and endpoint state manager."""
    redis_settings = config.general_settings
    redis_manager = RedisManager(
        host=redis_settings.redis_host,
        port=redis_settings.redis_port,
        password=redis_settings.redis_password or None,
        ssl_enabled=redis_settings.redis_ssl,
        ssl_cert_reqs=redis_settings.redis_ssl_cert_reqs,
    )

    await redis_manager.connect()
    redis_client = redis_manager.get_client()

    # Use a default state TTL of 2 hours (7200 seconds)
    endpoint_state_manager = EndpointStateManager(
        redis_client=redis_client,
        state_ttl=7200,  # 2 hours default
    )

    return redis_manager, endpoint_state_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global load_balancer, redis_manager, cache_manager, llm_client, endpoint_state_manager, config

    # Startup
    logger.info("application_startup")

    # Load configuration
    config_path = os.getenv("LLMPROXY_CONFIG", "llmproxy.yaml")
    config = await load_config_async(config_path)
    logger.info("configuration_loaded", config_path=config_path)

    # Initialize all resources; ensure partial cleanup on failure before yield
    try:
        # Initialize Redis and endpoint state manager
        redis_manager, endpoint_state_manager = await init_redis(config)
        logger.info("redis_initialized")

        # Initialize cache manager
        cache_ttl = 604800  # 7 days default
        if config.general_settings.cache_params:
            cache_ttl = config.general_settings.cache_params.ttl

        cache_manager = CacheManager(
            redis_client=redis_manager.get_client(),
            ttl=cache_ttl,
            namespace="llmproxy",
            cache_enabled=config.general_settings.cache,
        )
        logger.info("cache_manager_initialized")

        # Initialize LLM client
        llm_client = LLMClient(
            timeout=config.general_settings.http_timeout,
            max_connections=config.general_settings.http_max_connections,
        )
        logger.info("llm_client_initialized")

        # Initialize load balancer with endpoint state manager
        load_balancer = LoadBalancer(
            cooldown_time=config.general_settings.cooldown_time,
            allowed_fails=config.general_settings.allowed_fails,
            state_manager=endpoint_state_manager,
        )
        await load_balancer.initialize_from_config(config)
        logger.info("load_balancer_initialized")
    except Exception as e:
        logger.error("startup_initialization_failed", error=str(e))
        # Best-effort cleanup of partially initialized resources
        try:
            if llm_client:
                await llm_client.close()
                logger.info("llm_client_closed_after_failure")
        finally:
            if redis_manager:
                await redis_manager.disconnect()
        raise

    # Signal handlers are handled by uvicorn, no need for custom handlers

    yield

    # Cleanup
    logger.info("application_shutdown")
    if llm_client:
        # Ensure HTTP connection pools are closed
        await llm_client.close()
        logger.info("llm_client_closed")
    if redis_manager:
        # Use RedisManager's disconnect method for proper cleanup
        await redis_manager.disconnect()
    if load_balancer:
        # Finalize load balancer (no external resources, but keep logs consistent)
        await load_balancer.shutdown()


app = FastAPI(
    title="LLM Proxy",
    description="A load balancing proxy for LLM API endpoints",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""

    redis_connected = False
    state_sharing_enabled = False
    endpoints_configured = 0
    model_groups = []

    if redis_manager:
        redis_connected = await redis_manager.health_check()

    if endpoint_state_manager:
        state_sharing_enabled = await endpoint_state_manager.health_check()

    if load_balancer:
        model_groups = load_balancer.get_model_groups()
        # Count endpoints from configs
        for group_configs in load_balancer.endpoint_configs.values():
            endpoints_configured += len(group_configs)

    return HealthResponse(
        status="healthy" if redis_connected and state_sharing_enabled else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        redis_connected=redis_connected,
        state_sharing_enabled=state_sharing_enabled,
        endpoints_configured=endpoints_configured,
        model_groups=model_groups,
    )


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get endpoint statistics from Redis."""
    global load_balancer  # noqa: F824

    if not load_balancer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Load balancer not initialized",
        )

    try:
        # Get fresh stats directly from Redis
        endpoint_stats = await load_balancer.get_all_endpoints_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "model_groups": endpoint_stats,
            "total_model_groups": len(endpoint_stats),
            "total_endpoints": sum(len(group) for group in endpoint_stats.values()),
        }
    except Exception as e:
        logger.error("stats_fetch_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch statistics: {str(e)}",
        )


@app.exception_handler(LLMProxyError)
async def llm_proxy_exception_handler(
    request: Request, exc: LLMProxyError
) -> JSONResponse:
    """Handle LLMProxy-specific exceptions."""
    logger.error(
        "llm_proxy_error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.delete("/cache")
async def clear_cache() -> Dict[str, Any]:
    """Clear all cache entries (for testing)."""
    if not redis_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis not available",
        )

    try:
        if cache_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache manager not available",
            )

        # Delegate to CacheManager's safe implementation (SCAN + batched UNLINK/DEL)
        deleted_count = await cache_manager.invalidate_all()

        logger.info(
            "cache_cleared",
            deleted_entries=deleted_count,
            method="cache_manager.invalidate_all",
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "deleted_entries": deleted_count,
            "status": "success",
        }
    except Exception as e:
        logger.error("cache_clear_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


# Dependency functions that ensure non-None returns
def get_load_balancer_required() -> LoadBalancer:
    """Get load balancer with validation."""
    if load_balancer is None:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")
    return load_balancer


def get_cache_manager_required() -> CacheManager:
    """Get cache manager with validation."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    return cache_manager


def get_llm_client_required() -> LLMClient:
    """Get LLM client with validation."""
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    return llm_client


def get_config_required() -> Any:
    """Get config with validation."""
    if config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")
    return config


# Include all API routes without prefix (OpenAI compatible)
app.include_router(
    create_router(
        get_load_balancer=get_load_balancer_required,
        cache_manager=get_cache_manager_required,
        llm_client=get_llm_client_required,
        config=get_config_required,
    )
)


def get_load_balancer() -> Optional[LoadBalancer]:
    """Dependency injection for load balancer."""
    return load_balancer


def get_cache_manager() -> Optional[CacheManager]:
    """Dependency injection for cache manager."""
    return cache_manager


def get_llm_client() -> Optional[LLMClient]:
    """Dependency injection for LLM client."""
    return llm_client


def get_config() -> Any:
    """Dependency injection for config."""
    return config


def main() -> None:
    """Main entry point for uvicorn."""
    import uvicorn

    # Set up logging first (only if not already configured)
    if not logging.getLogger().hasHandlers():
        setup_logging()

    try:
        # Load configuration to get proper host and port
        config_path = os.getenv("LLMPROXY_CONFIG", "llmproxy.yaml")
        config = asyncio.run(load_config_async(config_path))

        # Use configured bind address and port
        host = config.general_settings.bind_address
        port = config.general_settings.bind_port

        logger.info("starting_server", host=host, port=port, config_path=config_path)

        uvicorn.run(
            "llmproxy.main:app",
            host=host,
            port=port,
            log_config=None,  # Use our custom logging
        )
    except KeyboardInterrupt:
        logger.info("received_keyboard_interrupt")
    except Exception as e:
        logger.error("application_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
