from typing import List, Optional, Union

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for an individual model endpoint"""

    model: str
    weight: int = 1
    params: dict = {}  # All OpenAI client parameters go here


class ModelGroup(BaseModel):
    """Configuration for a group of models"""

    model_group: str
    models: List[ModelConfig]

    model_config = {"protected_namespaces": ()}  # Disable protected namespace warning


class RedisConfig(BaseModel):
    """Redis configuration"""

    host: str
    port: Union[int, str]
    password: str = ""
    db: int = 0


class CacheParams(BaseModel):
    """Cache configuration parameters"""

    type: str = "redis"
    ttl: int = 604800  # 7 days default
    namespace: str = "llmproxy.cache"
    host: str
    port: Union[int, str]
    password: str


class GeneralSettings(BaseModel):
    """General configuration settings"""

    bind_address: str = "127.0.0.1"
    bind_port: int
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

    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration from general settings"""
        return RedisConfig(
            host=self.general_settings.redis_host,
            port=self.general_settings.redis_port,
            password=self.general_settings.redis_password,
            db=0,
        )

    model_config = {
        "protected_namespaces": (),  # Disable protected namespace warning
        "extra": "allow",  # Allow extra fields in case the YAML has additional properties
    }
