"""Tests for the error handler module"""

from unittest.mock import AsyncMock, patch

import pytest

from llmproxy.api.error_handler import (
    APIError,
    EndpointError,
    RateLimitError,
    RetryHandler,
    is_retryable_error,
)


class TestRetryHandler:
    """Test cases for RetryHandler class"""

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self):
        """Test successful execution on first attempt"""
        retry_handler = RetryHandler(max_retries=3)
        mock_func = AsyncMock(return_value="success")

        result = await retry_handler.execute_with_retry(
            mock_func, "arg1", kwarg1="value1"
        )

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_failures(self):
        """Test successful execution after some failures"""
        retry_handler = RetryHandler(max_retries=3, base_delay=0.01)
        mock_func = AsyncMock(
            side_effect=[Exception("fail"), Exception("fail"), "success"]
        )

        result = await retry_handler.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """Test retry exhaustion with all attempts failing"""
        retry_handler = RetryHandler(max_retries=2, base_delay=0.01)
        test_exception = Exception("persistent error")
        mock_func = AsyncMock(side_effect=test_exception)

        with pytest.raises(Exception, match="persistent error"):
            await retry_handler.execute_with_retry(mock_func)

        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_specific_exception_types(self):
        """Test retry only on specific exception types"""
        retry_handler = RetryHandler(max_retries=3, base_delay=0.01)
        specific_error = ValueError("specific error")
        mock_func = AsyncMock(side_effect=specific_error)

        with pytest.raises(ValueError):
            await retry_handler.execute_with_retry(mock_func, retry_on=(ValueError,))

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately"""
        retry_handler = RetryHandler(max_retries=3, base_delay=0.01)
        runtime_error = RuntimeError("runtime error")
        mock_func = AsyncMock(side_effect=runtime_error)

        with pytest.raises(RuntimeError):
            await retry_handler.execute_with_retry(mock_func, retry_on=(ValueError,))

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_exponential_backoff(self):
        """Test exponential backoff delay calculation"""
        retry_handler = RetryHandler(max_retries=3, base_delay=0.1, max_delay=1.0)
        mock_func = AsyncMock(side_effect=Exception("fail"))

        with patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(Exception):
                await retry_handler.execute_with_retry(mock_func)

            # Verify sleep was called with increasing delays
            assert mock_sleep.call_count == 2
            # First delay should be around base_delay (0.1) + jitter
            first_delay = mock_sleep.call_args_list[0][0][0]
            assert 0.1 <= first_delay <= 0.11

            # Second delay should be around 2 * base_delay (0.2) + jitter
            second_delay = mock_sleep.call_args_list[1][0][0]
            assert 0.2 <= second_delay <= 0.22

    @pytest.mark.asyncio
    async def test_execute_with_retry_max_delay_cap(self):
        """Test that delay is capped at max_delay"""
        retry_handler = RetryHandler(max_retries=5, base_delay=10.0, max_delay=2.0)
        mock_func = AsyncMock(side_effect=Exception("fail"))

        with patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(Exception):
                await retry_handler.execute_with_retry(mock_func)

            # All delays should be capped at max_delay
            for call in mock_sleep.call_args_list:
                delay = call[0][0]
                assert delay <= 2.2  # max_delay + 10% jitter

    def test_retry_handler_initialization(self):
        """Test RetryHandler initialization with custom parameters"""
        retry_handler = RetryHandler(max_retries=5, base_delay=2.0, max_delay=120.0)

        assert retry_handler.max_retries == 5
        assert retry_handler.base_delay == 2.0
        assert retry_handler.max_delay == 120.0

    def test_retry_handler_default_initialization(self):
        """Test RetryHandler initialization with default parameters"""
        retry_handler = RetryHandler()

        assert retry_handler.max_retries == 3
        assert retry_handler.base_delay == 1.0
        assert retry_handler.max_delay == 60.0


class TestAPIError:
    """Test cases for APIError class"""

    def test_api_error_initialization(self):
        """Test APIError initialization"""
        error = APIError("Test error message", status_code=400, error_type="test_error")

        assert error.message == "Test error message"
        assert error.status_code == 400
        assert error.error_type == "test_error"
        assert str(error) == "Test error message"

    def test_api_error_default_values(self):
        """Test APIError with default values"""
        error = APIError("Default error")

        assert error.message == "Default error"
        assert error.status_code == 500
        assert error.error_type == "api_error"

    def test_api_error_to_dict(self):
        """Test APIError to_dict method"""
        error = APIError("Test error", status_code=404, error_type="not_found")
        expected_dict = {
            "error": {
                "message": "Test error",
                "type": "not_found",
                "code": 404,
            }
        }

        assert error.to_dict() == expected_dict


class TestRateLimitError:
    """Test cases for RateLimitError class"""

    def test_rate_limit_error_initialization(self):
        """Test RateLimitError initialization"""
        error = RateLimitError("Rate limit exceeded", retry_after=60)

        assert error.message == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_type == "rate_limit_error"
        assert error.retry_after == 60

    def test_rate_limit_error_default_values(self):
        """Test RateLimitError with default values"""
        error = RateLimitError()

        assert error.message == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_type == "rate_limit_error"
        assert error.retry_after is None

    def test_rate_limit_error_to_dict(self):
        """Test RateLimitError to_dict method"""
        error = RateLimitError("Custom rate limit message", retry_after=120)
        expected_dict = {
            "error": {
                "message": "Custom rate limit message",
                "type": "rate_limit_error",
                "code": 429,
            }
        }

        assert error.to_dict() == expected_dict


class TestEndpointError:
    """Test cases for EndpointError class"""

    def test_endpoint_error_initialization(self):
        """Test EndpointError initialization"""
        error = EndpointError(
            "Endpoint failed", endpoint_id="endpoint-1", status_code=503
        )

        assert error.message == "Endpoint failed"
        assert error.status_code == 503
        assert error.error_type == "endpoint_error"
        assert error.endpoint_id == "endpoint-1"

    def test_endpoint_error_default_status_code(self):
        """Test EndpointError with default status code"""
        error = EndpointError("Endpoint error", endpoint_id="endpoint-2")

        assert error.message == "Endpoint error"
        assert error.status_code == 500
        assert error.endpoint_id == "endpoint-2"

    def test_endpoint_error_to_dict(self):
        """Test EndpointError to_dict method"""
        error = EndpointError(
            "Service unavailable", endpoint_id="endpoint-3", status_code=503
        )
        expected_dict = {
            "error": {
                "message": "Service unavailable",
                "type": "endpoint_error",
                "code": 503,
            }
        }

        assert error.to_dict() == expected_dict


class TestIsRetryableError:
    """Test cases for is_retryable_error function"""

    def test_retryable_status_codes(self):
        """Test that retryable status codes return True"""
        retryable_codes = [408, 429, 500, 501, 502, 503, 504, 599]

        for code in retryable_codes:
            assert is_retryable_error(code) is True

    def test_non_retryable_status_codes(self):
        """Test that non-retryable status codes return False"""
        non_retryable_codes = [200, 201, 400, 401, 403, 404, 405, 422, 499]

        for code in non_retryable_codes:
            assert is_retryable_error(code) is False

    def test_boundary_status_codes(self):
        """Test boundary status codes"""
        assert is_retryable_error(499) is False  # Just below 500
        assert is_retryable_error(500) is True  # Start of 5xx range
        assert is_retryable_error(599) is True  # End of 5xx range
        assert is_retryable_error(600) is False  # Just above 599
