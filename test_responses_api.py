#!/usr/bin/env python3
"""Test script for LLMProxy Responses API endpoint"""

import asyncio
import httpx
import json
from typing import Optional


async def test_responses_api(
    proxy_url: str = "http://localhost:8080",
    model: str = "gpt-4o-mini",
    stream: bool = False
):
    """Test the Responses API endpoint"""
    
    # Responses API uses 'input' instead of 'messages'
    request_data = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": "Write a one-sentence bedtime story about a unicorn."
            }
        ],
        "stream": stream
    }
    
    print(f"\n{'='*60}")
    print(f"Testing Responses API - Model: {model}, Streaming: {stream}")
    print(f"{'='*60}\n")
    
    async with httpx.AsyncClient() as client:
        try:
            if stream:
                # Handle streaming response
                async with client.stream(
                    "POST",
                    f"{proxy_url}/responses",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=60.0
                ) as response:
                    print(f"Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        print("\nStreaming response:")
                        print("-" * 40)
                        
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    print("\n[Stream completed]")
                                    break
                                
                                try:
                                    data = json.loads(data_str)
                                    # Extract content from the response
                                    if "type" in data and data["type"] == "message":
                                        if "content" in data:
                                            for content_item in data["content"]:
                                                if content_item.get("type") == "output_text":
                                                    print(content_item.get("text", ""), end="", flush=True)
                                except json.JSONDecodeError:
                                    print(f"Could not parse: {data_str}")
                    else:
                        error_text = await response.aread()
                        print(f"Error: {error_text.decode()}")
            else:
                # Handle non-streaming response
                response = await client.post(
                    f"{proxy_url}/responses",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=60.0
                )
                
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print("\nResponse:")
                    print("-" * 40)
                    
                    # Extract output_text from the response
                    if "output_text" in data:
                        print(f"Output: {data['output_text']}")
                    else:
                        # Pretty print the full response structure
                        print(json.dumps(data, indent=2))
                    
                    # Show proxy metadata if available
                    print("\nProxy Metadata:")
                    print(f"  Cache Hit: {data.get('_proxy_cache_hit', 'N/A')}")
                    print(f"  Latency: {data.get('_proxy_latency_ms', 'N/A')}ms")
                    if "_proxy_endpoint_base_url" in data:
                        print(f"  Endpoint: {data['_proxy_endpoint_base_url']}")
                else:
                    error = response.json()
                    print(f"Error: {json.dumps(error, indent=2)}")
                    
        except Exception as e:
            print(f"Request failed: {str(e)}")


async def test_responses_with_previous_id():
    """Test Responses API with previous_response_id for conversation continuity"""
    
    proxy_url = "http://localhost:8080"
    model = "gpt-4o-mini"
    
    print(f"\n{'='*60}")
    print("Testing Responses API with conversation continuity")
    print(f"{'='*60}\n")
    
    async with httpx.AsyncClient() as client:
        # First request
        first_request = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": "Hi! My name is Alice. Can you remember this?"
                }
            ],
            "stream": False
        }
        
        print("First request: Introducing myself as Alice")
        response1 = await client.post(
            f"{proxy_url}/responses",
            json=first_request,
            headers={"Content-Type": "application/json"},
            timeout=60.0
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            response_id = data1.get("id")
            print(f"Response ID: {response_id}")
            print(f"Assistant: {data1.get('output_text', 'N/A')}")
            
            # Second request using previous_response_id
            if response_id:
                second_request = {
                    "model": model,
                    "input": [
                        {
                            "role": "user",
                            "content": "What's my name?"
                        }
                    ],
                    "previous_response_id": response_id,
                    "stream": False
                }
                
                print("\nSecond request: Asking about my name")
                response2 = await client.post(
                    f"{proxy_url}/responses",
                    json=second_request,
                    headers={"Content-Type": "application/json"},
                    timeout=60.0
                )
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    print(f"Assistant: {data2.get('output_text', 'N/A')}")
                else:
                    print(f"Second request failed: {response2.json()}")
        else:
            print(f"First request failed: {response1.json()}")


async def main():
    """Run all tests"""
    
    # Test basic non-streaming request
    await test_responses_api(stream=False)
    
    # Test streaming request
    await test_responses_api(stream=True)
    
    # Test conversation continuity
    # Note: This feature depends on the upstream provider supporting it
    await test_responses_with_previous_id()


if __name__ == "__main__":
    asyncio.run(main()) 