# LLMProxy

A high-performance proxy server for Large Language Models (LLMs) that provides intelligent load balancing, automatic failover, rate limiting, and response caching.

## Features

- **Load Balancing**: Distribute requests across multiple LLM endpoints using weighted round-robin
- **Automatic Failover**: Seamlessly switch to backup endpoints when primary endpoints fail
- **Rate Limit Management**: Track and respect API rate limits to prevent 429 errors
- **Response Caching**: Cache deterministic responses in Redis for improved performance
- **Streaming Cache**: Cache and replay streaming responses for faster real-time interactions
- **OpenAI Compatible**: Drop-in replacement for OpenAI API clients (both Chat Completions and Responses APIs)
- **Multi-Provider Support**: Works with OpenAI and Azure OpenAI endpoints
- **Health Monitoring**: Track endpoint health and automatically cooldown failed endpoints
- **Responses API Support**: Full support for OpenAI's new Responses API with conversation state management

## Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- API keys for OpenAI and/or Azure OpenAI

### Installation

#### Option 1: Install from GitHub (Recommended)

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/llmproxy.git

# Or install in development mode
git clone https://github.com/yourusername/llmproxy.git
cd llmproxy
pip install -e .
```

#### Option 2: Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llmproxy.git
cd llmproxy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Create your configuration file:
```bash
# Copy the example configuration
cp llmproxy.yaml.example llmproxy.yaml
```

2. Set up environment variables:

**Option A: For testing (no real API keys needed)**
```bash
python setup_test_env.py
```

**Option B: For production use**
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

3. Edit `llmproxy.yaml` with your specific configuration (API keys, endpoints, etc.)

### Running

1. Start Redis (if not already running):
```bash
redis-server
```

2. Run the proxy:
```bash
# If installed via pip
llmproxy

# Or with custom config
llmproxy --config /path/to/your/llmproxy.yaml

# Or with custom host/port
llmproxy --host 0.0.0.0 --port 8080

# For development setup
python run_proxy.py
```

The proxy will start on the address and port configured in `llmproxy.yaml` (default: `http://127.0.0.1:5000`).

### Usage

Use any OpenAI-compatible client and point it to the proxy:

#### Chat Completions API

```python
from openai import OpenAI

# Point to the local proxy instead of OpenAI
# The proxy URL is configured in llmproxy.yaml
client = OpenAI(
    base_url="http://127.0.0.1:5000/v1",  # Match your llmproxy.yaml settings
    api_key="dummy-key"  # Auth is handled by proxy
)

# Use as normal
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Streaming with Caching

LLMProxy now supports caching for streaming responses, dramatically improving performance for repeated streaming requests.

You can also disable caching for specific requests:

```python
# Disable cache for this request
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What's the weather?"}],
    extra_body={"cache": {"no-cache": True}}  # Bypass cache
)
```

#### Responses API (New)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5000",  # No /v1 prefix for Responses API
    api_key="dummy-key"
)

# Use the new Responses API
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "Hello!"}]
)

# Access the response
print(response.output_text)
```

##### Streaming with Caching (Responses API)

The Responses API now also supports caching for streaming responses.

You can also disable caching for specific requests:

```python
# Disable cache for this request
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What's the weather?"}],
    extra_body={"cache": {"no-cache": True}}  # Bypass cache
)
```

### Configuration

Edit `llmproxy.yaml` to configure:

- Model groups and endpoints
- Weights for load distribution
- Rate limiting and retry settings
- Caching parameters
- Bind address and port (default: 127.0.0.1:5000)

See `llmproxy.yaml` for a complete example. The bind address and port are set in the `general_settings` section:

```yaml
general_settings:
  bind_address: 127.0.0.1
  bind_port: 5000
  # ... other settings
```

### API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `POST /responses` - OpenAI's new Responses API
- `GET /health` - Health check endpoint
- `GET /stats` - Proxy statistics and endpoint status

### Responses API Features

The proxy fully supports OpenAI's new Responses API, including:

- **Conversation State**: Use `previous_response_id` to maintain conversation context
- **Semantic Events**: Better streaming support with typed events
- **Future Tools**: Ready for web search, file search, and computer use tools
- **Structured Outputs**: Use `text.format` for structured responses

See [RESPONSES_API.md](RESPONSES_API.md) for detailed documentation on using the Responses API.

## Architecture

LLMProxy uses:
- **FastAPI** for high-performance async request handling
- **Redis** for caching and rate limit tracking
- **httpx** for async HTTP client requests
- **Pydantic** for configuration validation

## Testing

### Running Live Tests

The `test_proxy_live.py` script tests the proxy with real LLM requests:

```bash
# Run all tests
python test_proxy_live.py

# Run quick test only
python test_proxy_live.py --quick

# Test against a different proxy URL
python test_proxy_live.py --proxy-url http://localhost:8080
```

Tests include:
- Basic chat completions
- Streaming responses
- Caching behavior (including streaming cache)
- Error handling
- Load balancing
- Concurrent requests
- Health and stats endpoints

### Testing Streaming Cache

Run the streaming cache demo:

```bash
python demo_streaming_cache.py
```

This demonstrates how streaming responses are cached and replayed with dramatically reduced latency.

## Development

### Running Unit Tests

```bash
# Run tests (automatically runs in parallel with -n auto)
pytest

# Run tests serially (disable parallel execution)
pytest -n 0

# Run tests in parallel with specific number of workers
pytest -n 4

# For debugging cache-related issues, run serially
pytest tests/test_proxy_live_pytest.py -n 0 -v

# Run all tests except caching tests in parallel (default behavior)
pytest -k "not (caching or cache)"
```

### Code Formatting

```bash
black llmproxy/
flake8 llmproxy/
```

## License

MIT License - see LICENSE file for details