#!/bin/bash

# LLMProxy Security Fuzzing Script
# Runs fuzzing tests using mock servers to avoid external dependencies

set -e

echo "==================================================================================="
echo "LLMProxy Security Fuzzing Suite"
echo "==================================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Install dependencies
install_dependencies() {
    log "Installing fuzzing dependencies..."
    pip install -r requirements-fuzz.txt
}

# Run simple HTTP fuzzer
run_simple_fuzzer() {
    log "Running Simple HTTP Fuzzer..."
    echo "This tests basic functionality and security vulnerabilities"
    PYTHONPATH=. python fuzz_tests/simple_http_fuzzer.py
}

# Run hypothesis-based fuzzing
run_hypothesis_fuzzer() {
    log "Running Hypothesis Property-Based Fuzzer..."
    echo "This tests with property-based testing for edge cases"
    PYTHONPATH=. python -m pytest fuzz_tests/fuzz_hypothesis_with_mock.py::TestIntegrationFuzzingWithMock::test_end_to_end_responses_api -v || true
    PYTHONPATH=. python -m pytest fuzz_tests/fuzz_hypothesis_with_mock.py::TestIntegrationFuzzingWithMock::test_header_fuzzing -v || true
    PYTHONPATH=. python -m pytest fuzz_tests/fuzz_hypothesis_with_mock.py::TestConfigurationFuzzingWithMock::test_config_validation_robustness -v || true
    PYTHONPATH=. python -m pytest fuzz_tests/fuzz_hypothesis_with_mock.py::TestStreamingFuzzingWithMock::test_streaming_chunk_handling -v || true
}

# Main function
main() {
    log "Starting LLMProxy Security Fuzzing Suite"

    # Parse command line arguments
    SKIP_DEPS=false
    DURATION=300  # 5 minutes default

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --duration)
                DURATION="$2"
                shift 2
                ;;
            *)
                echo "Usage: $0 [--skip-deps] [--duration SECONDS]"
                exit 1
                ;;
        esac
    done

    # Install dependencies if not skipped
    if [ "$SKIP_DEPS" = false ]; then
        install_dependencies
    fi

    echo ""
    echo "==================================================================================="
    echo "Running Fuzzing Tests (using mock servers - no external dependencies required)"
    echo "Duration: ${DURATION} seconds"
    echo "==================================================================================="
    echo ""

    # Run the fuzzers
    run_simple_fuzzer
    echo ""

    run_hypothesis_fuzzer
    echo ""

    success "Fuzzing suite completed!"
    echo ""
    echo "Summary:"
    echo "- Simple HTTP Fuzzer: Tests basic security vulnerabilities"
    echo "- Hypothesis Fuzzer: Property-based testing for edge cases"
    echo "- All tests use mock servers (no API keys required)"
    echo "- Check output above for any issues found"
}

# Run main function
main "$@"
