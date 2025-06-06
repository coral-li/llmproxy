from typing import List, Optional, Union
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for an individual model endpoint"""
    model: str
    weight: int = 1
    params: dict = {}  # All OpenAI client parameters go here


class ModelGroup(BaseModel):
    """Configuration for a group of models"""
    model_group: str
    models: List[ModelConfig]


class CacheParams(BaseModel):
    """Cache configuration parameters"""
    type: str = "redis"
    ttl: int = 604800  # 7 days default
    namespace: str = "litellm.cache"
    host: str
    port: Union[int, str]
    password: str


class GeneralSettings(BaseModel):
    """General configuration settings"""
    bind_address: str = "127.0.0.1"
    bind_port: int = 4000
    num_retries: int = 3
    allowed_fails: int = 1
    cooldown_time: int = 60
    redis_host: str
    redis_port: Union[int, str]
    redis_password: str
    cache: bool = True
    cache_params: Optional[CacheParams] = None


class LLMProxyConfig(BaseModel):
    """Root configuration model for LLM Proxy"""
    model_groups: List[ModelGroup]
    general_settings: GeneralSettings

    class Config:
        # Allow extra fields in case the YAML has additional properties
        extra = "allow" 