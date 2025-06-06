import os
import time
import json
import hashlib
from typing import Any, Dict, List, Optional

import yaml
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config_model import LLMProxyConfig, ModelConfig
import redis.asyncio as redis


# ----- Configuration Loading -----


def resolve_env_vars(data: Any) -> Any:
    """Recursively resolve environment variable references."""
    missing_vars: List[str] = []

    def _resolve(item: Any) -> Any:
        if isinstance(item, dict):
            return {k: _resolve(v) for k, v in item.items()}
        if isinstance(item, list):
            return [_resolve(x) for x in item]
        if isinstance(item, str) and item.startswith("os.environ/"):
            env_var = item.replace("os.environ/", "")
            value = os.getenv(env_var)
            if value is None:
                missing_vars.append(env_var)
                return item
            # try to cast numbers
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        return item

    resolved = _resolve(data)

    if missing_vars:
        unique = sorted(set(missing_vars))
        msg = ", ".join(unique)
        raise ValueError(
            "The following environment variables are not set: " f"{msg}"
        )  # noqa: E501
    return resolved


def load_config(yaml_path: str) -> LLMProxyConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    data = resolve_env_vars(data)
    return LLMProxyConfig(**data)


# ----- Endpoint & Group Helpers -----


class Endpoint:
    def __init__(self, config: ModelConfig, eid: str):
        self.config = config
        self.id = eid


class WeightedGroup:
    def __init__(
        self,
        name: str,
        models: List[ModelConfig],
        settings: LLMProxyConfig.GeneralSettings,
        r: redis.Redis,
    ):
        self.name = name
        self.redis = r
        self.settings = settings
        self.main: List[Endpoint] = []
        self.fallback: List[Endpoint] = []
        counter = 0
        for m in models:
            ep = Endpoint(m, f"{name}-{counter}")
            counter += 1
            if m.weight > 0:
                self.main.extend([ep] * m.weight)
            else:
                self.fallback.append(ep)
        self.index = 0

    async def _is_cooled(self, ep: Endpoint) -> bool:
        key = f"cooldown:{ep.id}"
        val = await self.redis.get(key)
        if val is None:
            return True
        return float(val) <= time.time()

    async def _record_failure(self, ep: Endpoint) -> None:
        fail_key = f"fails:{ep.id}"
        count = await self.redis.incr(fail_key)
        if count >= self.settings.allowed_fails:
            cool_key = f"cooldown:{ep.id}"
            await self.redis.setex(
                cool_key,
                self.settings.cooldown_time,
                str(time.time() + self.settings.cooldown_time),
            )
            await self.redis.delete(fail_key)

    async def _reset_failures(self, ep: Endpoint) -> None:
        await self.redis.delete(f"fails:{ep.id}")

    async def get_next_endpoint(self) -> Optional[Endpoint]:
        checked = 0
        total = len(self.main)
        if total:
            while checked < total:
                ep = self.main[self.index]
                self.index = (self.index + 1) % total
                if await self._is_cooled(ep):
                    return ep
                checked += 1
        for ep in self.fallback:
            if await self._is_cooled(ep):
                return ep
        return None


# ----- LLM Proxy Server -----


class LLMProxy:
    def __init__(self, config: LLMProxyConfig):
        self.config = config
        self.app = FastAPI()
        self.redis = redis.Redis(
            host=config.general_settings.redis_host,
            port=int(config.general_settings.redis_port),
            password=config.general_settings.redis_password,
            decode_responses=True,
        )
        self.groups: Dict[str, WeightedGroup] = {}
        for group in config.model_groups:
            wg = WeightedGroup(
                group.model_group,
                group.models,
                config.general_settings,
                self.redis,
            )
            self.groups[group.model_group] = wg
        self.client = httpx.AsyncClient(timeout=30)
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            data = await request.json()
            model = data.get("model")
            if not model:
                raise HTTPException(
                    status_code=400,
                    detail="model is required",
                )
            group = self.groups.get(model)
            if not group:
                raise HTTPException(
                    status_code=400,
                    detail="Unknown model",
                )

            cache_key = None
            temp_zero = data.get("temperature", 0) == 0
            if self.config.general_settings.cache and temp_zero:
                key_raw = json.dumps(data, sort_keys=True)
                hashed = hashlib.sha256(key_raw.encode()).hexdigest()
                cache_key = f"cache:{model}:{hashed}"
                cached = await self.redis.get(cache_key)
                if cached:
                    return JSONResponse(
                        status_code=200,
                        content=json.loads(cached),
                    )

            attempt = 0
            max_attempts = self.config.general_settings.num_retries
            last_error = None
            while attempt < max_attempts:
                ep = await group.get_next_endpoint()
                if not ep:
                    break
                try:
                    response = await self._forward(ep, data)
                    if response.status_code < 300:
                        await group._reset_failures(ep)
                        content = response.json()
                        if cache_key:
                            ttl = (
                                self.config.general_settings.cache_params.ttl
                                if self.config.general_settings.cache_params
                                else 604800
                            )
                            await self.redis.setex(
                                cache_key,
                                ttl,
                                json.dumps(content),
                            )
                        return JSONResponse(
                            status_code=response.status_code, content=content
                        )
                    else:
                        await group._record_failure(ep)
                        last_error = (
                            response.status_code,
                            await response.aread(),
                        )
                except Exception as e:
                    await group._record_failure(ep)
                    last_error = e
                attempt += 1
            if isinstance(last_error, tuple):
                status, body = last_error
                raise HTTPException(status_code=status, detail=body.decode())
            raise HTTPException(status_code=503, detail="All endpoints failed")

    async def _forward(
        self,
        ep: Endpoint,
        payload: Dict[str, Any],
    ) -> httpx.Response:
        params = ep.config.params
        base_url = params.get("base_url", "https://api.openai.com")
        headers = {"Content-Type": "application/json"}
        api_key = params.get("api_key")
        if api_key:
            if "azure" in base_url:
                headers["api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        url = base_url.rstrip("/") + "/v1/chat/completions"
        query = params.get("default_query", {})
        return await self.client.post(
            url,
            headers=headers,
            params=query,
            json=payload,
        )


async def create_app(config_path: str) -> FastAPI:
    config = load_config(config_path)
    proxy = LLMProxy(config)
    return proxy.app


if __name__ == "__main__":
    cfg_path = os.environ.get("LLMPROXY_CONFIG", "llmproxy.yaml")
    config = load_config(cfg_path)
    proxy = LLMProxy(config)
    import uvicorn

    uvicorn.run(
        proxy.app,
        host=config.general_settings.bind_address,
        port=config.general_settings.bind_port,
    )
