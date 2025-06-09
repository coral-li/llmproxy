# LLMProxy

## Problem statement

### Load balancing
In production applications that are making heavy use of LLMs, we need the ability to distribute the load between endpoints. This includes balancing between different Azure regions that might have different rate limits as well as distributing some of the load to the official OpenAI API.

### Reliability
Azure endpoints in particular often have outages and spotty reliability. Handling these errors with retries and fallback to other endpoints in client code introduces complexity and code maintainbility issues.

### Cost management
Some providers offer lower prices while offering lower rate limits and lower reliability. That means we need a way to send as much LLM traffic to the cheaper providers while respecting rate limits and handling 429 errors and outages internally.


## Features

* Intended to run on localhost so that the application can just make all LLM calls to localhost
* Implements the OpenAI API and forwards all requests to upstream LLM APIs
* Upstream LLM APIs can include OpenAI API, OpenAI Azure (by providing api_key, base_url and default_query={"api-version": "preview"})
* Balancing load between multiple upstream endpoints
   - Upstream endpoints should support configurable weighting
* Keep track of and respect rate limits returned in the header of each response
* Gracefully handle 429 rate limit errors and retry request with a different endpoint
* Cooldown of overloaded or degraded endpoints to prevent unnecessary retries
* Support for multiple models (e.g. gpt-4.1 and o4-mini) with the ability to define a range of upstream endpoints for each of them (model groups)
* Simple configuration of model groups with weight, base_url, api_keys etc. in a configuration file
   - Ability to read api_keys and other configuration parameters from the environment
* Caching of LLM responses in redis
* High performance, fully production-ready, async proxy server
* Support the core functionality of the OpenAI API
   - Chat Completions API
   - New Responses API

## Configuration

### Config Pydantic Model

```python
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

```

### Example config
```yaml
model_groups:
  ### o4-mini
  # Azure (default, distributed across 2 endpoints with weight 1)
  - model_group: o4-mini
    models:
      - model: o4-mini
        weight: 1
        params:
          api_key: os.environ/AZURE_OPENAI_API_KEY_O4_MINI_1
          base_url: os.environ/AZURE_OPENAI_ENDPOINT_O4_MINI_1
          default_query:
            api-version: "preview"

      - model: o4-mini
        weight: 1
        params:
          api_key: os.environ/AZURE_OPENAI_API_KEY_O4_MINI_2
          base_url: os.environ/AZURE_OPENAI_ENDPOINT_O4_MINI_2
          default_query:
            api-version: "preview"

      # OpenAI (fallback with weight 0, only gets traffic if all other endpoints are cooled down)
      - model: o4-mini
        weight: 0
        params:
          api_key: os.environ/OPENAI_API_KEY

  ### gpt-4.1
  # Azure (default, distributed across 2 endpoints with weight 1)
  - model_group: gpt-4.1
    models:
      - model: gpt-4.1
        weight: 1
        params:
          api_key: os.environ/AZURE_OPENAI_API_KEY_GPT_4_POINT_1_1
          base_url: os.environ/AZURE_OPENAI_ENDPOINT_GPT_4_POINT_1_1
          default_query:
            api-version: "preview"

      - model: gpt-4.1
        weight: 1
        params:
          api_key: os.environ/AZURE_OPENAI_API_KEY_GPT_4_POINT_1_2
          base_url: os.environ/AZURE_OPENAI_ENDPOINT_GPT_4_POINT_1_2
          default_query:
            api-version: "preview"

      # OpenAI (fallback)
      - model: gpt-4.1
        weight: 0
        params:
          api_key: os.environ/OPENAI_API_KEY

general_settings:
  bind_address: 127.0.0.1
  bind_port: 4000
  num_retries: 3 # number of retries for each request
  allowed_fails: 1 # number of allowed fails for each endpoint
  cooldown_time: 60 # cooldown time for each endpoint

  # Redis will be used to store the cache and the state of the endpoints
  redis_host: os.environ/REDIS_HOST
  redis_port: os.environ/REDIS_PORT
  redis_password: os.environ/REDIS_PASSWORD

  cache: True # enable cache
  cache_params:        # set cache params for redis
    type: redis
    ttl: 604800 # 7 days
    namespace: "litellm.cache"
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
    password: os.environ/REDIS_PASSWORD
```

### Loading Config from YAML with env substitution

```python
import yaml
from config_model import LLMProxyConfig
import os

def load_config_from_yaml(yaml_file_path: str) -> LLMProxyConfig:
    """
    Load and validate YAML configuration using the Pydantic model

    Args:
        yaml_file_path: Path to the YAML configuration file

    Returns:
        Validated LLMProxyConfig instance
    """
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Replace environment variable references with actual values
    yaml_data = resolve_env_vars(yaml_data)

    # Create and validate the configuration
    config = LLMProxyConfig(**yaml_data)
    return config


def resolve_env_vars(data):
    """
    Recursively resolve environment variable references in the format 'os.environ/VAR_NAME'
    Raises ValueError if any environment variable is not set.
    """
    missing_vars = []

    def _resolve(data):
        if isinstance(data, dict):
            return {key: _resolve(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [_resolve(item) for item in data]
        elif isinstance(data, str) and data.startswith('os.environ/'):
            env_var = data.replace('os.environ/', '')
            env_value = os.getenv(env_var)
            if env_value is None:
                missing_vars.append(env_var)
                return data  # Return original for now
            return env_value
        else:
            return data

    resolved_data = _resolve(data)

    if missing_vars:
        unique_missing = sorted(set(missing_vars))
        raise ValueError(f"The following environment variables are not set: {', '.join(unique_missing)}")

    return resolved_data
```

## Resources

https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-lifecycle?tabs=key#api-evolution
