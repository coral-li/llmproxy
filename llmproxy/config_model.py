from typing import List, Optional, Union

from pydantic import BaseModel, Field

DEFAULT_MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_CACHE_ENTRY_BYTES = 64 * 1024 * 1024


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


class UsageStreamParams(BaseModel):
    """Redis Stream sink for per-request usage telemetry.

    Omitting the whole `usage_stream` block leaves telemetry off, so existing
    deployments keep their current behaviour without touching their config.
    """

    enabled: bool = True
    # Consumers read this key with a consumer group; see docs/usage-telemetry.md.
    stream_key: str = "llmproxy:usage"
    # Approximate cap (XADD MAXLEN ~) so a stalled consumer cannot grow the
    # stream without bound. Redis trims to roughly this many entries.
    max_len: int = Field(default=1_000_000, gt=0)
    # Inbound request headers copied onto each record so a caller can attribute
    # a request to the agent or workflow that issued it.
    caller_headers: List[str] = Field(
        default_factory=lambda: [
            "x-coral-agent",
            "x-coral-run-id",
            "x-coral-feature",
        ]
    )


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
    usage_stream: Optional[UsageStreamParams] = None
    response_affinity_ttl: int = Field(default=21600, gt=0)
    max_request_body_bytes: int = Field(
        default=DEFAULT_MAX_REQUEST_BODY_BYTES,
        gt=0,
    )
    max_cache_entry_bytes: int = Field(
        default=DEFAULT_MAX_CACHE_ENTRY_BYTES,
        gt=0,
    )


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
