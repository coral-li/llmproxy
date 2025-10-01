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
- **Responses API Support**: Support for OpenAI's new Responses API
- **Comprehensive test suite with automated setup**: Tests automatically start and stop proxy instances

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
pip install -e .
```

### Configuration

1. Create your configuration file:
```bash
# Copy the example configuration
cp llmproxy.yaml.example llmproxy.yaml
```

2. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

3. Edit `llmproxy.yaml` with your specific configuration (API keys, endpoints, etc.)

### Running

1. Start Redis (if not already running):
```bash
# macOS
brew services start redis

# Linux
sudo systemctl start redis

# Or manually
redis-server
```

2. Run the proxy:
```bash
python -m llmproxy.cli
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
    base_url="http://127.0.0.1:5000",  # Match your llmproxy.yaml settings
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
- Bind address and port (configurable via `llmproxy.yaml`)

See `llmproxy.yaml` for a complete example. The bind address and port are set in the `general_settings` section:

```yaml
general_settings:
  bind_address: 127.0.0.1
  bind_port: 5000
  http_timeout: 300  # seconds
  # ... other settings
```

### API Endpoints

- `POST /chat/completions` - OpenAI-compatible chat completions
- `POST /responses` - OpenAI's new Responses API
- `GET /health` - Health check endpoint
- `GET /stats` - Proxy statistics and endpoint status


## Architecture

LLMProxy uses:
- **FastAPI** for high-performance async request handling
- **Redis** for caching and rate limit tracking
- **httpx** for async HTTP client requests
- **Pydantic** for configuration validation

## Testing

The test suite now automatically starts and stops proxy instances, eliminating the need to manually start the proxy before running tests.

### Prerequisites

- The `redis-server` binary must be installed (tests will start a temporary
  instance if Redis is not already running on port 6379)
- The `requests` package (included in pyproject.toml dependencies)

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_proxy_live_pytest.py::TestProxyLive::test_basic_completion -v

# Run tests with minimal output
python -m pytest tests/ -v --tb=no
```

### Test Features

- **Automated server management**: Tests automatically start mock LLM endpoints and proxy instances
- **Session-scoped fixtures**: Server startup/shutdown happens once per test session for efficiency
- **Isolated test environments**: Each test run uses random ports to avoid conflicts
- **Redis requirement validation**: Tests fail fast if Redis is not available
- **Comprehensive coverage**: Tests cover basic completions, streaming, caching, load balancing, error handling, and more

### Test Architecture

The test setup includes:

1. **Mock OpenAI servers**: Lightweight FastAPI servers that simulate OpenAI API endpoints
2. **Automated proxy startup**: Uses the actual proxy application with test configuration
3. **Cache management**: Automatic cache clearing between test sessions
4. **Health monitoring**: Waits for services to be ready before running tests

### Configuration

Tests use `llmproxy.test.yaml` which configures:
- Mock endpoints on localhost:8001-8003
- Redis caching with test namespace
- Reduced timeouts for faster test execution

## Configuration

See `llmproxy.yaml.example` for full configuration options.

## Development

The automated test setup makes development much easier:

1. No need to manually start/stop services
2. Tests run in isolation with their own proxy instances
3. Automatic cleanup prevents port conflicts
4. Fast feedback loop for development

## Requirements

- Python 3.8+
- Redis server
- See `pyproject.toml` for Python dependencies

## Contributing

1. Ensure Redis is running
2. Run the test suite: `python -m pytest tests/`
3. All tests should pass before submitting changes

The automated test setup ensures consistent testing across different development environments.

## License

MIT License - see LICENSE file for details
