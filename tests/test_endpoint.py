"""Tests for the endpoint model"""

import hashlib
import json

from llmproxy.models.endpoint import Endpoint, EndpointStatus


class TestEndpointStatus:
    """Test cases for EndpointStatus enum"""

    def test_endpoint_status_values(self):
        """Test EndpointStatus enum values"""
        assert EndpointStatus.HEALTHY.value == "healthy"
        assert EndpointStatus.DEGRADED.value == "degraded"
        assert EndpointStatus.COOLING_DOWN.value == "cooling_down"


class TestEndpoint:
    """Test cases for Endpoint class"""

    def test_endpoint_initialization_basic(self):
        """Test basic endpoint initialization"""
        params = {"api_key": "test-key", "base_url": "https://api.openai.com"}
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=params)

        assert endpoint.model == "gpt-3.5-turbo"
        assert endpoint.weight == 1
        assert endpoint.params == params
        assert endpoint.allowed_fails == 1  # default value
        assert endpoint.base_url == "https://api.openai.com"
        assert endpoint.is_azure is False
        assert endpoint.id is not None
        assert len(endpoint.id) == 16  # SHA256 hash truncated to 16 chars

    def test_endpoint_initialization_with_allowed_fails(self):
        """Test endpoint initialization with custom allowed_fails"""
        params = {"api_key": "test-key"}
        endpoint = Endpoint(model="gpt-4", weight=2, params=params, allowed_fails=3)

        assert endpoint.allowed_fails == 3

    def test_endpoint_azure_detection(self):
        """Test Azure endpoint detection"""
        azure_params = {
            "api_key": "test-key",
            "base_url": "https://myresource.openai.azure.com",
        }
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=azure_params)

        assert endpoint.is_azure is True
        assert endpoint.base_url == "https://myresource.openai.azure.com"

    def test_endpoint_azure_detection_case_insensitive(self):
        """Test Azure endpoint detection is case insensitive"""
        azure_params = {
            "api_key": "test-key",
            "base_url": "https://myresource.openai.AZURE.com",
        }
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=azure_params)

        assert endpoint.is_azure is True

    def test_endpoint_non_azure_detection(self):
        """Test non-Azure endpoint detection"""
        openai_params = {"api_key": "test-key", "base_url": "https://api.openai.com"}
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=openai_params)

        assert endpoint.is_azure is False

    def test_endpoint_default_base_url(self):
        """Test endpoint with no base_url defaults to 'openai'"""
        params = {"api_key": "test-key"}
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=params)

        assert endpoint.base_url == "openai"
        assert endpoint.is_azure is False

    def test_endpoint_non_string_base_url(self):
        """Test endpoint with non-string base_url"""
        params = {"api_key": "test-key", "base_url": 12345}  # Non-string value
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=params)

        assert endpoint.base_url == 12345
        assert endpoint.is_azure is False

    def test_deterministic_id_generation(self):
        """Test that ID generation is deterministic"""
        params1 = {"api_key": "test-key", "base_url": "https://api.openai.com"}
        params2 = {
            "api_key": "test-key",  # Same API key
            "base_url": "https://api.openai.com",
        }

        endpoint1 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params1)
        endpoint2 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params2)

        # IDs should be the same because all parameters are identical
        assert endpoint1.id == endpoint2.id

        # Test that different API keys produce different IDs
        params3 = {
            "api_key": "different-key",  # Different API key
            "base_url": "https://api.openai.com",
        }
        endpoint3 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params3)

        # IDs should be different because API key is included in ID generation
        assert endpoint1.id != endpoint3.id

    def test_deterministic_id_different_models(self):
        """Test that different models generate different IDs"""
        params = {"api_key": "test-key", "base_url": "https://api.openai.com"}

        endpoint1 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params)
        endpoint2 = Endpoint(model="gpt-4", weight=1, params=params)

        assert endpoint1.id != endpoint2.id

    def test_deterministic_id_different_base_urls(self):
        """Test that different base URLs generate different IDs"""
        params1 = {"api_key": "test-key", "base_url": "https://api.openai.com"}
        params2 = {"api_key": "test-key", "base_url": "https://api.anthropic.com"}

        endpoint1 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params1)
        endpoint2 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params2)

        assert endpoint1.id != endpoint2.id

    def test_deterministic_id_with_azure_params(self):
        """Test ID generation with Azure-specific parameters"""
        params1 = {
            "api_key": "test-key",
            "base_url": "https://myresource.openai.azure.com",
            "deployment": "gpt-35-turbo",
            "api_version": "2024-02-15-preview",
        }
        params2 = {
            "api_key": "test-key",  # Same API key
            "base_url": "https://myresource.openai.azure.com",
            "deployment": "gpt-35-turbo",
            "api_version": "2024-02-15-preview",
        }

        endpoint1 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params1)
        endpoint2 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params2)

        # Should have same ID because all parameters are identical
        assert endpoint1.id == endpoint2.id

        # Test that different API keys produce different IDs
        params3 = {
            "api_key": "different-key",
            "base_url": "https://myresource.openai.azure.com",
            "deployment": "gpt-35-turbo",
            "api_version": "2024-02-15-preview",
        }
        endpoint3 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params3)

        # Should have different ID due to different API key
        assert endpoint1.id != endpoint3.id

    def test_deterministic_id_different_deployments(self):
        """Test that different Azure deployments generate different IDs"""
        params1 = {
            "api_key": "test-key",
            "base_url": "https://myresource.openai.azure.com",
            "deployment": "gpt-35-turbo",
            "api_version": "2024-02-15-preview",
        }
        params2 = {
            "api_key": "test-key",
            "base_url": "https://myresource.openai.azure.com",
            "deployment": "gpt-4",
            "api_version": "2024-02-15-preview",
        }

        endpoint1 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params1)
        endpoint2 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params2)

        assert endpoint1.id != endpoint2.id

    def test_deterministic_id_excludes_none_values(self):
        """Test that None values are excluded from ID generation"""
        params1 = {
            "api_key": "test-key",
            "base_url": "https://api.openai.com",
            "deployment": None,
            "api_version": None,
        }
        params2 = {"api_key": "test-key", "base_url": "https://api.openai.com"}

        endpoint1 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params1)
        endpoint2 = Endpoint(model="gpt-3.5-turbo", weight=1, params=params2)

        # Should have same ID because None values are excluded
        assert endpoint1.id == endpoint2.id

    def test_get_config_dict(self):
        """Test get_config_dict method"""
        params = {"api_key": "test-key", "base_url": "https://api.openai.com"}
        endpoint = Endpoint(
            model="gpt-3.5-turbo", weight=2, params=params, allowed_fails=3
        )

        config_dict = endpoint.get_config_dict()

        expected_dict = {
            "id": endpoint.id,
            "model": "gpt-3.5-turbo",
            "weight": 2,
            "base_url": "https://api.openai.com",
            "is_azure": False,
            "allowed_fails": 3,
        }

        assert config_dict == expected_dict

    def test_get_config_dict_azure(self):
        """Test get_config_dict method for Azure endpoint"""
        params = {
            "api_key": "test-key",
            "base_url": "https://myresource.openai.azure.com",
        }
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=params)

        config_dict = endpoint.get_config_dict()

        assert config_dict["is_azure"] is True
        assert config_dict["base_url"] == "https://myresource.openai.azure.com"

    def test_repr(self):
        """Test string representation of endpoint"""
        params = {"api_key": "test-key", "base_url": "https://api.openai.com"}
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=params)

        repr_str = repr(endpoint)

        assert "Endpoint" in repr_str
        assert "gpt-3.5-turbo" in repr_str
        assert "https://api.openai.com" in repr_str

    def test_id_generation_algorithm(self):
        """Test the specific algorithm used for ID generation"""
        params = {
            "api_key": "test-key",
            "base_url": "https://api.openai.com",
            "deployment": "test-deployment",
            "api_version": "2024-02-15",
        }
        endpoint = Endpoint(model="gpt-3.5-turbo", weight=1, params=params)

        # Manually calculate expected ID (now includes API key)
        id_data = {
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com",
            "deployment": "test-deployment",
            "api_version": "2024-02-15",
            "api_key": "test-key",
        }
        id_str = json.dumps(id_data, sort_keys=True)
        expected_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

        assert endpoint.id == expected_id
