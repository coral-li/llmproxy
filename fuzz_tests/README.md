# LLMProxy Fuzzing Tests with Mock Servers

This directory contains comprehensive fuzzing tests for LLMProxy that use mock servers instead of real upstream providers. This approach provides several advantages:

- **No external dependencies**: Tests work without real OpenAI API keys
- **Consistent results**: Mock servers provide predictable responses
- **No rate limits**: Can run intensive fuzzing without hitting API limits
- **Cost-free**: No API usage charges
- **Faster execution**: No network latency to external services

## Available Fuzzing Approaches

### 1. Simple HTTP Fuzzer (`simple_http_fuzzer.py`)
**Recommended for first-time users**

A lightweight fuzzer that requires only standard Python libraries:
- Basic security testing (injection attempts, malformed JSON, etc.)
- Robustness testing (large payloads, concurrent requests)
- Unicode and encoding tests
- No external fuzzing frameworks required

**Run it:**
```bash
cd fuzz_tests
python simple_http_fuzzer.py
```

### 2. Advanced Atheris Fuzzer (`fuzz_with_mock_server.py`)
**For advanced users with coverage-guided fuzzing**

Uses Atheris for sophisticated coverage-guided fuzzing:
- Systematic input generation
- Coverage tracking
- Advanced payload generation

**Requirements:**
```bash
pip install atheris requests
```

**Run it:**
```bash
cd fuzz_tests
python fuzz_with_mock_server.py
```

### 3. Property-Based Testing (`fuzz_hypothesis_with_mock.py`)
**For systematic edge case testing**

Uses Hypothesis for property-based testing:
- Systematic edge case generation
- Property invariant testing
- Component-level testing

**Requirements:**
```bash
pip install hypothesis pytest pytest-asyncio
```

**Run it:**
```bash
cd fuzz_tests
python -m pytest fuzz_hypothesis_with_mock.py -v
```

## Quick Start

### Prerequisites
1. **Redis**: `brew services start redis` (macOS) or `sudo systemctl start redis` (Linux)
2. **Python 3.8+**: Standard installation

### Run Simple Fuzzing (Recommended)
```bash
# From the project root
cd fuzz_tests
python simple_http_fuzzer.py
```

This will:
1. Start mock OpenAI servers
2. Start LLMProxy configured to use the mock servers
3. Run comprehensive security and robustness tests
4. Clean up automatically

### Run All Fuzzing Tests
```bash
# From the project root
./run_fuzzing.sh
```

This runs all available fuzzing approaches and generates a comprehensive report.

## Mock Server Architecture

The fuzzing tests use the same mock infrastructure as the existing test suite:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Fuzzer        │───▶│   LLMProxy       │───▶│ Mock OpenAI      │
│   (Test Client) │    │   (Under Test)   │    │ Servers          │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │   Redis Cache    │
                       │   (Real)         │
                       └──────────────────┘
```

### Mock Server Features
- **Multiple endpoints**: Simulates load balancing scenarios
- **Predictable responses**: Consistent behavior for testing
- **Error simulation**: Can simulate various failure modes
- **Streaming support**: Tests both regular and streaming responses
- **OpenAI compatibility**: Implements both Chat Completions and Responses APIs

## Test Categories

### Security Tests
- **Injection attacks**: SQL injection, command injection, XSS
- **Path traversal**: Directory traversal attempts
- **Header manipulation**: Malformed HTTP headers
- **Payload poisoning**: Malicious request payloads

### Robustness Tests
- **Large payloads**: Memory exhaustion attempts
- **Malformed JSON**: Invalid request structures
- **Missing fields**: Incomplete requests
- **Invalid types**: Wrong data types
- **Unicode handling**: Special characters and encodings

### Performance Tests
- **Concurrent requests**: Load testing
- **Memory leaks**: Resource usage monitoring
- **Response times**: Performance validation

### Functional Tests
- **API compliance**: OpenAI API compatibility
- **Error handling**: Proper error responses
- **Cache behavior**: Caching functionality
- **Load balancing**: Endpoint selection logic

## Interpreting Results

### Success Criteria
- **No crashes** (HTTP 500 errors)
- **Graceful error handling** (appropriate 4xx responses)
- **No memory leaks**
- **Consistent performance**

### Common Issues to Look For
1. **Server crashes**: 500 errors indicating unhandled exceptions
2. **Memory growth**: Steadily increasing memory usage
3. **Slow responses**: Performance degradation
4. **Invalid responses**: Malformed JSON or incorrect structure

### Example Output
```
Running fuzzing tests against http://127.0.0.1:54321
============================================================
✓ Basic chat completions
✓ Health endpoint
✓ Stats endpoint
✓ Responses API
✓ Large payload handling
✓ Malformed JSON handling
✓ Missing required fields
✓ Invalid model names
✓ Injection attempts
✓ Unicode handling
✓ Concurrent requests
============================================================
Test Results:
  Total tests: 11
  Passed: 11 ✓
  Failed: 0 ✗
  Crashes: 0 💥

🎉 All tests passed! No critical issues found.
```

## Troubleshooting

### Redis Connection Issues
```bash
# Check if Redis is running
redis-cli ping

# Start Redis (macOS)
brew services start redis

# Start Redis (Linux)
sudo systemctl start redis
```

### Port Conflicts
The fuzzing tests automatically find free ports, but if you see binding errors:
- Stop other instances of LLMProxy
- Check for processes using ports 6379 (Redis) and high-numbered ports

### Memory Issues
If tests fail with memory errors:
- Reduce payload sizes in the test configuration
- Monitor system memory usage
- Consider running tests on a machine with more RAM

### Import Errors
If you see import errors for test modules:
- Run from the project root directory
- Ensure the project is properly installed: `pip install -e .`

## Customization

### Adding New Tests
1. Add test methods to `SimpleFuzzer` class in `simple_http_fuzzer.py`
2. Follow the pattern: `def test_your_test_name(self):`
3. Use assertions to validate behavior
4. Register the test in `run_all_tests()`

### Modifying Mock Responses
Edit the mock server responses in `tests/conftest.py`:
- `MockOpenAIServer._select_response_content()`
- Add new response patterns for your test cases

### Configuring Test Scenarios
Modify the test configuration in the fuzzer classes:
- Adjust payload sizes
- Add new injection patterns
- Configure timeout values

## Contributing

When adding new fuzzing tests:
1. Use the mock server infrastructure
2. Test for security vulnerabilities
3. Ensure tests clean up properly
4. Add documentation for new test categories
5. Follow the existing patterns for consistency

## References

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [Atheris Fuzzing](https://github.com/google/atheris)
- [Hypothesis Property-Based Testing](https://hypothesis.readthedocs.io/)
- [RESTler API Fuzzing](https://github.com/microsoft/restler-fuzzer)
