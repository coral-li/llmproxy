from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for an individual model endpoint"""

    model: str
    weight: int = 1
    # Use default_factory to avoid shared mutable default across instances
    params: dict = Field(default_factory=dict)


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
    http_timeout: float = Field(default=300.0, gt=0)
    http_max_connections: int = Field(default=100, gt=0)
    num_retries: int = 3
    allowed_fails: int = 1
    cooldown_time: int = 60
    redis_host: str
    redis_port: Union[int, str]
    redis_password: str
    redis_ssl: bool = False
    redis_ssl_cert_reqs: Optional[str] = None  # Options: "required", "optional", "none"
    cache: bool = True
    cache_params: Optional[CacheParams] = None
    response_affinity_ttl: int = Field(default=21600, gt=0)


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
