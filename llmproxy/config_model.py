import re
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_CACHE_ENTRY_BYTES = 64 * 1024 * 1024
#: Default prefix of response-cache keys, `cache_params.namespace`. `DELETE
#: /cache` removes everything under the configured prefix, so nothing that has
#: to survive a cache clear may be stored there.
RESPONSE_CACHE_NAMESPACE = "llmproxy"
#: Header names that carry credentials. A caller header is kept in the usage
#: stream for as long as its record is, so none of these may be listed.
_CREDENTIAL_HEADER = re.compile(
    r"authorization|cookie|api[-_]?key|token|secret|password", re.IGNORECASE
)


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
    # Prefix of every cached response's key, and what clearing the cache sweeps.
    namespace: str = RESPONSE_CACHE_NAMESPACE
    host: str
    port: Union[int, str]
    password: str


class UsageStreamParams(BaseModel):
    """Redis Stream sink for per-request usage telemetry.

    Omitting the whole `usage_stream` block leaves telemetry off, so existing
    deployments keep their current behaviour without touching their config.
    """

    enabled: bool = True
    # Consumers read this key from a position they track themselves; see
    # docs/usage-telemetry.md.
    stream_key: str = "llmproxy-telemetry:usage"
    # Approximate cap (XADD MAXLEN ~). Redis never evicts a stream and reading
    # one does not shrink it, so the stream settles at about this many entries:
    # at roughly 0.75 KB each, the default holds about 75 MB.
    max_len: int = Field(default=100_000, gt=0)
    # Inbound request headers copied onto each record so a caller can attribute
    # a request to the agent or workflow that issued it.
    caller_headers: List[str] = Field(default_factory=list)

    @field_validator("caller_headers")
    @classmethod
    def _no_credential_headers(cls, names: List[str]) -> List[str]:
        credentials = [name for name in names if _CREDENTIAL_HEADER.search(name)]
        if credentials:
            raise ValueError(
                "usage_stream.caller_headers must not list a header that carries "
                f"credentials, which the stream would keep: {', '.join(credentials)}"
            )
        return names


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

    @property
    def cache_namespace(self) -> str:
        """The key prefix of cached responses, which clearing the cache sweeps."""
        if self.cache_params is None:
            return RESPONSE_CACHE_NAMESPACE
        return self.cache_params.namespace

    @model_validator(mode="after")
    def _usage_stream_outside_the_cache(self) -> "GeneralSettings":
        if self.usage_stream and self.usage_stream.stream_key.startswith(
            f"{self.cache_namespace}:"
        ):
            raise ValueError(
                "usage_stream.stream_key must not start with "
                f"'{self.cache_namespace}:', where clearing the response cache "
                "would delete it"
            )
        return self


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
