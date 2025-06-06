#!/usr/bin/env python
"""Run LLMProxy server"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # Import necessary modules
    from llmproxy.main import app
    from llmproxy.config.config_loader import load_config
    import uvicorn

    # Load configuration to get bind address and port
    config = load_config()
    host = config.general_settings.bind_address
    port = config.general_settings.bind_port
    
    print(f"Starting LLMProxy on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    ) 