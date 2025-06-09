import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llmproxy.api.routes import create_router
from llmproxy.clients.llm_client import LLMClient
from llmproxy.config.config_loader import load_config
from llmproxy.core.cache_manager import CacheManager
from llmproxy.core.exceptions import LLMProxyError
from llmproxy.core.logger import get_logger, setup_logging
from llmproxy.core.redis_manager import RedisManager
from llmproxy.managers.endpoint_state_manager import EndpointStateManager
from llmproxy.managers.load_balancer import LoadBalancer

logger = get_logger(__name__)

# Global state
load_balancer: Optional[LoadBalancer] = None
redis_client: Optional[redis.Redis] = None
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


async def init_redis(config: Any) -> Tuple[redis.Redis, EndpointStateManager]:
    """Initialize Redis connection and endpoint state manager."""
    redis_settings = config.general_settings
    redis_manager = RedisManager(
        host=redis_settings.redis_host,
        port=redis_settings.redis_port,
        password=redis_settings.redis_password or None,
    )

    await redis_manager.connect()
    redis_client = redis_manager.get_client()

    # Use a default state TTL of 2 hours (7200 seconds)
    endpoint_state_manager = EndpointStateManager(
        redis_client=redis_client,
        state_ttl=7200,  # 2 hours default
    )

    return redis_client, endpoint_state_manager


async def shutdown_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger.info("received_shutdown_signal", signal=signum)

    # Perform cleanup here
    if redis_client:
        # Use aclose() to properly close the async Redis client
        await redis_client.aclose()

    sys.exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global load_balancer, redis_client, cache_manager, llm_client, endpoint_state_manager, config

    # Startup
    logger.info("application_startup")

    # Load configuration
    config_path = os.getenv("LLMPROXY_CONFIG", "llmproxy.yaml")
    config = load_config(config_path)
    logger.info("configuration_loaded", config_path=config_path)

    # Initialize Redis and endpoint state manager
    try:
        redis_client, endpoint_state_manager = await init_redis(config)
        logger.info("redis_initialized")
    except Exception as e:
        logger.error("redis_initialization_failed", error=str(e))
        raise

    # Initialize cache manager
    cache_ttl = 604800  # 7 days default
    if config.general_settings.cache_params:
        cache_ttl = config.general_settings.cache_params.ttl

    cache_manager = CacheManager(
        redis_client=redis_client,
        ttl=cache_ttl,
        namespace="llmproxy",
        cache_enabled=config.general_settings.cache,
    )
    logger.info("cache_manager_initialized")

    # Initialize LLM client
    llm_client = LLMClient()
    logger.info("llm_client_initialized")

    # Initialize load balancer with endpoint state manager
    load_balancer = LoadBalancer(
        cooldown_time=config.general_settings.cooldown_time,
        allowed_fails=config.general_settings.allowed_fails,
        state_manager=endpoint_state_manager,
    )
    await load_balancer.initialize_from_config(config)
    logger.info("load_balancer_initialized")

    # Setup signal handlers for graceful shutdown (only in main thread)
    try:
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(shutdown_handler(sig, None))
            )
    except RuntimeError as e:
        # Signal handlers can only be set in the main thread
        logger.info("signal_handlers_skipped", reason=str(e))

    yield

    # Cleanup
    logger.info("application_shutdown")
    if redis_client:
        # Use aclose() to properly close the async Redis client
        await redis_client.aclose()


app = FastAPI(
    title="LLM Proxy",
    description="A load balancing proxy for LLM API endpoints",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    global load_balancer, redis_client, endpoint_state_manager  # noqa: F824

    redis_connected = False
    state_sharing_enabled = False
    endpoints_configured = 0
    model_groups = []

    if redis_client:
        try:
            await redis_client.ping()
            redis_connected = True
        except Exception:
            pass

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
    if not redis_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis not available",
        )

    try:
        # Get all keys in the cache namespace
        keys = await redis_client.keys("llmproxy:*")

        deleted_count = 0
        if keys:
            deleted_count = await redis_client.delete(*keys)

        logger.info("cache_cleared", deleted_entries=deleted_count)

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


# Include API routes with /v1 prefix
app.include_router(
    create_router(
        get_load_balancer=lambda: load_balancer or LoadBalancer(),
        cache_manager=get_cache_manager_required,
        llm_client=get_llm_client_required,
        config=get_config_required,
    ),
    prefix="/v1",
)

# Also include routes without prefix for direct access
app.include_router(
    create_router(
        get_load_balancer=lambda: load_balancer or LoadBalancer(),
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

    # Set up logging first
    setup_logging()

    try:
        uvicorn.run(
            "llmproxy.main:app",
            host="0.0.0.0",
            port=4243,
            log_config=None,  # Use our custom logging
        )
    except KeyboardInterrupt:
        logger.info("received_keyboard_interrupt")
    except Exception as e:
        logger.error("application_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
