#!/usr/bin/env python
"""Example usage of LLMProxy with OpenAI client"""

from openai import OpenAI, AsyncOpenAI
import asyncio
import httpx
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
from llmproxy.config.utils import get_proxy_base_url, get_proxy_url


def sync_example():
    """Synchronous example using OpenAI client"""
    # Create client pointing to LLMProxy
    client = OpenAI(
        base_url=get_proxy_base_url(),
        api_key="dummy",  # LLMProxy handles the actual API keys
    )
    
    # Use any model configured in your proxy
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # This will be routed based on your config
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print("Sync response:", response.choices[0].message.content)


async def async_example():
    """Asynchronous example using AsyncOpenAI client"""
    # Create async client pointing to LLMProxy
    client = AsyncOpenAI(
        base_url=get_proxy_base_url(),
        api_key="dummy",  # LLMProxy handles the actual API keys
    )
    
    # Stream response
    stream = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short joke."}
        ],
        temperature=0.7,
        max_tokens=100,
        stream=True
    )
    
    print("Streaming response: ", end="")
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    print()


async def health_check():
    """Check proxy health"""
    proxy_url = get_proxy_url()
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{proxy_url}/health")
        if response.status_code == 200:
            print("\nHealth check:", response.json())
        else:
            print(f"Health check failed: {response.status_code}")


async def get_stats():
    """Get proxy statistics"""
    proxy_url = get_proxy_url()
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{proxy_url}/stats")
        if response.status_code == 200:
            print("\nProxy stats:", response.json())
        else:
            print(f"Stats request failed: {response.status_code}")


async def main():
    """Run all examples"""
    print("LLMProxy Examples\n")
    
    # Health check
    await health_check()
    
    # Run sync example
    print("\n--- Sync Example ---")
    sync_example()
    
    # Run async example
    print("\n--- Async Example ---")
    await async_example()
    
    # Get stats
    await get_stats()


if __name__ == "__main__":
    proxy_url = get_proxy_url()
    print(f"Using LLMProxy at {proxy_url}")
    print(f"Make sure LLMProxy is running on {proxy_url}")
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        print(f"Make sure LLMProxy is running!") 