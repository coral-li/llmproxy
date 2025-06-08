"""Exception classes for LLMProxy"""


class LLMProxyError(Exception):
    """Base exception class for LLMProxy"""
    pass


class ConfigurationError(LLMProxyError):
    """Raised when there are configuration errors"""
    pass


class EndpointError(LLMProxyError):
    """Raised when there are endpoint-related errors"""
    pass


class LoadBalancerError(LLMProxyError):
    """Raised when there are load balancer errors"""
    pass


class StateManagerError(LLMProxyError):
    """Raised when there are state manager errors"""
    pass


class RedisError(LLMProxyError):
    """Raised when there are Redis connection or operation errors"""
    pass


class CacheError(LLMProxyError):
    """Raised when there are cache-related errors"""
    pass 