from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import os
from typing import Optional
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llmproxy.config.config_loader import load_config
from llmproxy.managers.load_balancer import LoadBalancer
from llmproxy.core.cache_manager import CacheManager
from llmproxy.core.redis_manager import RedisManager
from llmproxy.core.logger import get_logger, setup_logging
from llmproxy.clients.llm_client import LLMClient
from llmproxy.api.chat_completions import ChatCompletionHandler
from llmproxy.api.responses import ResponseHandler

# Setup logging
setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# Global instances
config = None
load_balancer = None
cache_manager = None
llm_client = None
redis_manager = None
chat_handler = None
response_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global config, load_balancer, cache_manager, llm_client, redis_manager, chat_handler, response_handler

    try:
        # Load environment variables from .env file
        load_dotenv()
        logger.info("loaded_env_file")
        
        config = load_config()

        # Initialize Redis
        logger.info("initializing_redis")
        redis_manager = RedisManager(
            host=config.general_settings.redis_host,
            port=config.general_settings.redis_port,
            password=config.general_settings.redis_password,
        )
        await redis_manager.connect()

        # Initialize components
        logger.info("initializing_components")

        # Load balancer
        load_balancer = LoadBalancer(
            cooldown_time=config.general_settings.cooldown_time,
            allowed_fails=config.general_settings.allowed_fails,
        )

        # Cache manager
        cache_params = config.general_settings.cache_params
        if cache_params:
            ttl = cache_params.ttl
            namespace = cache_params.namespace
        else:
            ttl = 604800  # 7 days default
            namespace = "llmproxy"

        cache_manager = CacheManager(
            redis_manager.get_client(), ttl=ttl, namespace=namespace,
            cache_enabled=config.general_settings.cache
        )

        # LLM client
        llm_client = LLMClient()

        # Initialize endpoint pools
        await load_balancer.initialize_from_config(config)

        # Create chat handler
        chat_handler = ChatCompletionHandler(
            load_balancer=load_balancer,
            cache_manager=cache_manager,
            llm_client=llm_client,
            config=config,
        )

        # Create response handler
        response_handler = ResponseHandler(
            load_balancer=load_balancer,
            cache_manager=cache_manager,
            llm_client=llm_client,
            config=config,
        )

        logger.info(
            "startup_complete",
            model_groups=load_balancer.get_model_groups(),
            bind_address=config.general_settings.bind_address,
            bind_port=config.general_settings.bind_port,
        )

        yield

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    finally:
        # Cleanup on shutdown
        logger.info("shutdown_started")

        if llm_client:
            await llm_client.close()

        if redis_manager:
            await redis_manager.disconnect()

        logger.info("shutdown_complete")


# Create FastAPI app
app = FastAPI(
    title="LLMProxy",
    version="1.0.0",
    description="High-performance LLM proxy with load balancing, caching, and rate limiting",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions in OpenAI-compatible format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "api_error",
                "code": exc.status_code,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500,
            }
        },
    )


@app.post("/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint"""
    if not chat_handler:
        raise HTTPException(503, "Service not ready")

    try:
        request_data = await request.json()
        response = await chat_handler.handle_request(request_data)
        
        # If it's already a StreamingResponse, return it directly
        if isinstance(response, StreamingResponse):
            return response
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_completion_error", error=str(e))
        raise HTTPException(500, str(e))


@app.post("/responses")
async def responses(request: Request):
    """OpenAI-compatible responses API endpoint"""
    if not response_handler:
        raise HTTPException(503, "Service not ready")

    try:
        request_data = await request.json()
        response = await response_handler.handle_request(request_data)
        
        # If it's already a StreamingResponse, return it directly
        if isinstance(response, StreamingResponse):
            return response
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("response_api_error", error=str(e))
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    if not redis_manager:
        return {"status": "starting", "redis": "not_connected"}

    redis_healthy = await redis_manager.health_check()

    return {
        "status": "healthy" if redis_healthy else "degraded",
        "redis": "connected" if redis_healthy else "disconnected",
        "model_groups": load_balancer.get_model_groups() if load_balancer else [],
        "cache_stats": cache_manager.get_stats() if cache_manager else {},
    }


@app.get("/stats")
async def stats():
    """Get proxy statistics"""
    if not load_balancer:
        raise HTTPException(503, "Service not ready")

    return {
        "endpoints": load_balancer.get_all_endpoints_stats(),
        "cache": cache_manager.get_stats() if cache_manager else {},
    }


@app.delete("/cache")
async def invalidate_cache():
    """Invalidate all cached entries"""
    if not cache_manager:
        raise HTTPException(503, "Service not ready")
    
    try:
        deleted_count = await cache_manager.invalidate_all()
        logger.info("cache_invalidated_via_api", deleted_count=deleted_count)
        
        return {
            "message": "Cache invalidated successfully",
            "deleted_entries": deleted_count
        }
    except Exception as e:
        logger.error("cache_invalidation_api_error", error=str(e))
        raise HTTPException(500, f"Cache invalidation failed: {str(e)}")


@app.delete("/cache/pattern/{pattern}")
async def invalidate_cache_by_pattern(pattern: str):
    """Invalidate cached entries matching a pattern"""
    if not cache_manager:
        raise HTTPException(503, "Service not ready")
    
    try:
        deleted_count = await cache_manager.invalidate_by_pattern(pattern)
        logger.info("cache_invalidated_by_pattern_via_api", pattern=pattern, deleted_count=deleted_count)
        
        return {
            "message": f"Cache entries matching pattern '{pattern}' invalidated successfully",
            "pattern": pattern,
            "deleted_entries": deleted_count
        }
    except Exception as e:
        logger.error("cache_invalidation_pattern_api_error", error=str(e), pattern=pattern)
        raise HTTPException(500, f"Cache invalidation failed: {str(e)}")


@app.post("/cache/invalidate")
async def invalidate_cache_for_request(request: Request):
    """Invalidate cache for a specific request"""
    if not cache_manager:
        raise HTTPException(503, "Service not ready")
    
    try:
        request_data = await request.json()
        success = await cache_manager.invalidate_request(request_data)
        
        if success:
            logger.info("cache_invalidated_for_request_via_api", model=request_data.get("model"))
            return {
                "message": "Cache invalidated for request successfully",
                "request_cached": True
            }
        else:
            return {
                "message": "No cache entry found for request",
                "request_cached": False
            }
            
    except Exception as e:
        logger.error("cache_invalidation_request_api_error", error=str(e))
        raise HTTPException(500, f"Cache invalidation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration to get bind address and port
    from llmproxy.config.config_loader import load_config
    main_config = load_config()
    host = main_config.general_settings.bind_address
    port = main_config.general_settings.bind_port

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_config=None,  # Use our custom logging
    )
