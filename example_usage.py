import yaml
from config_model import LLMProxyConfig
import os
from dotenv import load_dotenv
from typing import Dict, Any
import sys

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


def resolve_env_vars(data: Any) -> Any:
    """
    Recursively resolve environment variable references in the format 'os.environ/VAR_NAME'
    Raises ValueError if any environment variable is not set.
    """
    missing_vars = []
    
    def _resolve(data: Any) -> Any:
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
            # Try to convert to int/float if possible
            try:
                return int(env_value)
            except ValueError:
                try:
                    return float(env_value)
                except ValueError:
                    return env_value
        else:
            return data
    
    resolved_data = _resolve(data)
    
    if missing_vars:
        unique_missing = sorted(set(missing_vars))
        raise ValueError(f"The following environment variables are not set: {', '.join(unique_missing)}")
    
    return resolved_data


def validate_config(config: LLMProxyConfig) -> None:
    """
    Perform additional validation on the configuration
    """
    # Check that we have at least one model group
    if not config.model_groups:
        raise ValueError("At least one model group must be defined")
    
    # Check that each model group has at least one model
    for group in config.model_groups:
        if not group.models:
            raise ValueError(f"Model group '{group.model_group}' must have at least one model")
        
        # Warn if all models have weight 0
        total_weight = sum(model.weight for model in group.models)
        if total_weight == 0:
            print(f"WARNING: Model group '{group.model_group}' has all models with weight 0")
    
    # Validate Redis configuration
    if config.general_settings.cache and config.general_settings.cache_params:
        if not all([
            config.general_settings.cache_params.host,
            config.general_settings.cache_params.port
        ]):
            raise ValueError("Redis host and port must be provided when caching is enabled")


def print_config_summary(config: LLMProxyConfig) -> None:
    """
    Print a summary of the loaded configuration
    """
    print("\n=== LLMProxy Configuration Summary ===")
    print(f"\nServer: {config.general_settings.bind_address}:{config.general_settings.bind_port}")
    print(f"Cache: {'Enabled' if config.general_settings.cache else 'Disabled'}")
    print(f"Retries: {config.general_settings.num_retries}")
    print(f"Cooldown: {config.general_settings.cooldown_time}s")
    
    print("\nModel Groups:")
    for group in config.model_groups:
        print(f"\n  {group.model_group}:")
        for model in group.models:
            weight_info = f"weight={model.weight}"
            if model.weight == 0:
                weight_info += " (fallback only)"
            
            provider = "OpenAI"
            if 'base_url' in model.params and 'azure' in model.params.get('base_url', '').lower():
                provider = "Azure OpenAI"
            
            print(f"    - {model.model} ({provider}, {weight_info})")


if __name__ == "__main__":
    try:
        # Load environment variables
        load_dotenv()
        
        # Load the configuration
        config = load_config_from_yaml("llmproxy.yaml")
        
        # Validate configuration
        validate_config(config)
        
        print("✅ Configuration loaded and validated successfully!")
        
        # Print configuration summary
        print_config_summary(config)
        
        # Optional: Export full config as JSON for debugging
        if "--export-json" in sys.argv:
            import json
            with open("config_export.json", "w") as f:
                json.dump(config.model_dump(), f, indent=2)
            print("\n📄 Configuration exported to config_export.json")
                
    except FileNotFoundError:
        print("❌ Error: llmproxy.yaml not found")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1) 