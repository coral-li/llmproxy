import sys
import os
import signal
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import llmproxy modules
from llmproxy.config.config_loader import load_config
from llmproxy.api.routes import create_router
from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.managers.endpoint_state_manager import EndpointStateManager
from llmproxy.core.cache_manager import CacheManager
from llmproxy.clients.llm_client import LLMClient
from llmproxy.core.logger import get_logger, setup_logging
from llmproxy.core.exceptions import LLMProxyError

logger = get_logger(__name__)

# Global state
load_balancer: Optional[LoadBalancer] = None
redis_client: Optional[aioredis.Redis] = None
endpoint_state_manager: Optional[EndpointStateManager] = None
cache_manager: Optional[CacheManager] = None
llm_client: Optional[LLMClient] = None
config: Optional[Any] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    redis_connected: bool
    state_sharing_enabled: bool
    endpoints_configured: int
    model_groups: list


async def init_redis(config) -> tuple[aioredis.Redis, EndpointStateManager]:
    """Initialize Redis connection and state manager"""
    redis_config = config.redis
    
    try:
        # Connect to Redis
        redis_conn = aioredis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            decode_responses=True,
        )

        # Test connection
        await redis_conn.ping()
        logger.info("redis_connected", host=redis_config.host, port=redis_config.port, db=redis_config.db)

        # Initialize state manager with 2-hour TTL
        state_manager = EndpointStateManager(redis_conn, state_ttl=2 * 60 * 60)

        # Test state manager
        is_healthy = await state_manager.health_check()
        if not is_healthy:
            raise Exception("State manager health check failed")

        logger.info("endpoint_state_manager_initialized", ttl_hours=2)
        return redis_conn, state_manager

    except Exception as e:
        logger.error("redis_initialization_failed", error=str(e))
        raise


async def shutdown_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("shutdown_signal_received", signal=signum)
    
    global load_balancer, redis_client
    
    if load_balancer:
        await load_balancer.shutdown()
    
    if redis_client:
        await redis_client.close()
    
    logger.info("application_shutdown_complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global load_balancer, redis_client, endpoint_state_manager, cache_manager, llm_client, config

    # Initialize
    try:
        config = load_config()
        
        # Initialize Redis and state manager
        redis_client, endpoint_state_manager = await init_redis(config)
        
        # Initialize cache manager
        cache_manager = CacheManager(redis_client, namespace="llmproxy")
        
        # Initialize LLM client with default timeout
        llm_client = LLMClient(timeout=6000.0)  # 100 minutes timeout
        
        # Initialize load balancer with state management
        load_balancer = LoadBalancer(
            cooldown_time=config.general_settings.cooldown_time,
            allowed_fails=config.general_settings.allowed_fails,
            state_manager=endpoint_state_manager,
        )
        
        # Initialize from config (will set up Redis state)
        await load_balancer.initialize_from_config(config)
        
        logger.info("application_started_successfully")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(shutdown_handler(s, f)))
        signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(shutdown_handler(s, f)))
        
        yield
        
    except Exception as e:
        logger.error("application_startup_failed", error=str(e))
        raise
    finally:
        # Cleanup
        if load_balancer:
            await load_balancer.shutdown()
        if llm_client:
            await llm_client.close()
        if redis_client:
            await redis_client.close()
        logger.info("application_cleanup_complete")


# Create FastAPI app
app = FastAPI(
    title="LLM Proxy",
    description="A load-balancing proxy for LLM APIs with Redis state sharing",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    global load_balancer, redis_client, endpoint_state_manager
    
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
    """Get endpoint statistics from Redis"""
    global load_balancer
    
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
async def llm_proxy_exception_handler(request: Request, exc: LLMProxyError):
    """Handle LLMProxy-specific exceptions"""
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
    """Clear all cache entries (for testing)"""
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
            "status": "success"
        }
    except Exception as e:
        logger.error("cache_clear_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


# Include API routes with /v1 prefix
app.include_router(
    create_router(
        get_load_balancer=lambda: load_balancer,
        cache_manager=lambda: cache_manager,
        llm_client=lambda: llm_client,
        config=lambda: config
    ),
    prefix="/v1"
)

# Also include routes without prefix for direct access
app.include_router(
    create_router(
        get_load_balancer=lambda: load_balancer,
        cache_manager=lambda: cache_manager,
        llm_client=lambda: llm_client,
        config=lambda: config
    )
)


def main():
    """Main entry point for uvicorn"""
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
