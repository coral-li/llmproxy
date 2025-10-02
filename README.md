# LLMProxy

LLMProxy is a FastAPI-based proxy that load balances across multiple Large Language Model (LLM) providers. It handles failover, retries, caching, and rate limits for you while remaining fully compatible with the OpenAI API surface.

## Why LLMProxy?

- **Provider-agnostic**: Register OpenAI, Azure OpenAI, and any OpenAI-compatible endpoints in a single configuration.
- **Graceful failover**: Automatically detect failing upstreams, cool them down, and retry requests against healthy endpoints.
- **Observability built in**: Health and statistics endpoints expose live state; Redis-backed state tracking keeps multiple proxy instances in sync.
- **Deterministic caching**: Cache both regular and streaming responses in Redis with fine-grained controls and manual cache invalidation.
- **Drop-in OpenAI compatibility**: Reuse existing SDK clients (chat completions, responses, embeddings) by only changing the base URL.
- **Battle-tested test suite**: End-to-end pytest harness spins up mock upstreams, Redis, and the proxy itself for reliable CI.

## Project Layout

```
llmproxy/
  api/          # OpenAI-compatible route handlers (chat, responses, embeddings)
  clients/      # Async HTTP client with streaming + retry helpers
  core/         # Caching, Redis, logging utilities
  managers/     # Load balancer + endpoint state management
  config/       # Pydantic config models & YAML loader
tests/          # Pytest suite with mock upstream servers
```

## Quick Start

### Requirements

- Python 3.9 or newer (3.11+ recommended)
- Redis 6+
- API keys for your upstream providers (OpenAI, Azure OpenAI, etc.)

### 1. Clone and install

```bash
git clone https://github.com/yourusername/llmproxy.git
cd llmproxy
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development tooling (black, mypy, etc.):

```bash
pip install -e ".[dev]"
pre-commit install
```

### 2. Configure credentials

```bash
cp llmproxy.yaml.example llmproxy.yaml
cp env.example .env
```

Fill in provider keys and Redis connection info in `.env`, then update `llmproxy.yaml` to point at your upstream endpoints. Configuration supports `os.environ/VARNAME` references so secrets can stay in the environment.

Key sections in `llmproxy.yaml`:

- `model_groups`: group endpoints that serve the same model (weights control routing bias).
- `general_settings`: bind address/port, retry/cooldown behavior, Redis connection.
- `cache_params`: optional cache override settings (namespace, TTL, custom Redis host).

### 3. Run the proxy

Start Redis if you do not have one running already (`brew services start redis` on macOS). Then launch the proxy:

```bash
llmproxy --config llmproxy.yaml
# or
python -m llmproxy.cli --log-level INFO
```

By default the proxy reads `llmproxy.yaml` in the working directory and binds to the address/port defined under `general_settings` (`127.0.0.1:4243` in the sample file).

Environment-based config: set `LLMPROXY_CONFIG=/path/to/config.yaml` to point the proxy at another file.

## Using LLMProxy

Any OpenAI SDK can target the proxy by swapping the base URL. Authentication to upstream providers is handled by the proxy.

### Chat Completions

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:4243",
    api_key="dummy-key",  # ignored by the proxy
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello from LLMProxy"}],
)

print(response.choices[0].message.content)
```

### Responses API + Streaming Cache

```python
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for event in response:
    # Streaming events are cached; repeated requests replay immediately
    print(event)
```

Disable caching per-request by passing `extra_body={"cache": {"no-cache": True}}`.

### Embeddings

```python
embeddings = client.embeddings.create(
    model="text-embedding-3-large",
    input="Searchable content",
)

vector = embeddings.data[0].embedding
```

### Administrative Endpoints

- `GET /health`: readiness info, upstream counts, Redis state.
- `GET /stats`: live per-endpoint statistics pulled from Redis.
- `DELETE /cache`: invalidate cached responses (useful for testing).

## Configuration Deep Dive

`llmproxy/config_model.py` defines the schema enforced at load time. Highlights:

- Weighted round-robin routing per `model_group` with Azure/OpenAI parameters stored under `params`.
- Redis-backed endpoint state shared across processes via `EndpointStateManager`.
- Tunable retry/cooldown thresholds (`allowed_fails`, `cooldown_time`, `num_retries`).
- Optional dedicated cache namespace and TTL overrides.

The async loader in `llmproxy/config/config_loader.py` resolves `os.environ/VAR` references and merges missing cache fields from general Redis settings.

## Architecture Overview

1. `FastAPI` app (`llmproxy/main.py`) wires the lifespan events, initializes Redis, cache, LLM client, and the load balancer.
2. `LoadBalancer` selects an endpoint per request, tracking health stats in Redis so multiple instances can share state.
3. `LLMClient` issues upstream HTTP/streaming requests with retries, respecting per-endpoint configuration.
4. `CacheManager` stores deterministic responses in Redis and replays streaming content from cached chunks.
5. Request handlers in `llmproxy/api` expose OpenAI-compatible endpoints (`/chat/completions`, `/responses`, `/embeddings`).

## Development Workflow

```bash
make install-dev   # editable install + dev dependencies + pre-commit
make pre-commit    # format, lint, type-check
make test          # run full pytest suite (starts mock servers + Redis)
```

Pytest spins up mock upstream servers, a dedicated proxy instance, and handles Redis automatically. Tests cover caching behavior, failover logic, streaming, and CLI ergonomics.

Useful scripts:

- `clear_cache.sh`: quick helper to hit the `/cache` endpoint locally.
- `make test-cov`: generate HTML coverage in `htmlcov/`.

## Contributing

1. Install dev dependencies and enable pre-commit.
2. Make your changes with accompanying tests.
3. Run `make pre-commit` and `make test` before opening a PR.

We welcome improvements to additional providers, caching strategies, and observability tooling.

## License

MIT License - see `LICENSE` for details.
