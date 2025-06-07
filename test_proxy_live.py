#!/usr/bin/env python
"""Live testing script for LLMProxy with real LLM requests"""

import asyncio
import time
import json
from openai import OpenAI
import httpx
from typing import List, Dict, Any
import hashlib
from datetime import datetime
from test_proxy_config import get_configured_model
import sys
from pathlib import Path
from dotenv import load_dotenv
# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
from llmproxy.config.utils import get_proxy_url, get_proxy_base_url
load_dotenv()

class ProxyTester:
    """Test harness for LLMProxy"""
    
    def __init__(self, base_url: str = None):
        """Initialize tester with proxy URL"""
        if base_url is None:
            base_url = get_proxy_url()
        self.base_url = base_url
        self.client = OpenAI(
            base_url=self.base_url,
        )
        self.results = []
        # Get a configured model from the proxy
        self.model = get_configured_model(base_url)
        self.log(f"Using model: {self.model}")
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {level}: {message}")
    
    def test_basic_completion(self) -> Dict[str, Any]:
        """Test basic chat completion"""
        self.log("Testing basic chat completion...")
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {"role": "user", "content": "Say 'Hello from LLMProxy!' and nothing else."}
                ],
                extra_body={"cache": {"no-cache": True}},
            )
            
            duration = time.time() - start_time
            content = response.choices[0].message.content
            
            self.log(f"✅ Success: {content} (took {duration:.2f}s)")
            
            # Check if response has proxy metadata
            cache_hit = getattr(response, '_proxy_cache_hit', None)
            latency = getattr(response, '_proxy_latency_ms', None)
            
            return {
                "test": "basic_completion",
                "success": True,
                "duration": duration,
                "response": content,
                "cache_hit": cache_hit,
                "proxy_latency_ms": latency
            }
            
        except Exception as e:
            self.log(f"❌ Failed: {str(e)}", "ERROR")
            return {
                "test": "basic_completion",
                "success": False,
                "error": str(e)
            }
    
    def test_streaming(self) -> Dict[str, Any]:
        """Test streaming response"""
        self.log("Testing streaming response...")
        
        start_time = time.time()
        try:
            self.log("Creating streaming request...")
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Count from 1 to 5, one number at a time."}
                ],
                stream=True,
                extra_body={"cache": {"no-cache": True}},
            )
                        
            chunks = []
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                assert hasattr(chunk, 'choices') and len(chunk.choices) > 0, "No choices or empty choices"

                assert hasattr(chunk.choices[0], 'delta'), "No delta in choices[0]"

                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            
            self.log(f"Stream iteration completed. Total chunks processed: {chunk_count}")
            
            duration = time.time() - start_time
            full_response = "".join(chunks)
            
            # Check if we received the expected content (numbers 1-5)
            # Different models may chunk differently, so we check for content not chunk count
            has_all_numbers = all(str(i) in full_response for i in range(1, 6))
            
            # We should have received at least one chunk with content
            min_chunks_required = 1
            success = len(chunks) >= min_chunks_required and has_all_numbers
            
            if success:
                self.log(f"✅ Streaming success: {len(chunks)} chunks received with all numbers 1-5 (took {duration:.2f}s)")
            else:
                if not has_all_numbers:
                    missing_numbers = [str(i) for i in range(1, 6) if str(i) not in full_response]
                    self.log(f"❌ Streaming failed: Missing numbers {missing_numbers} in response")
                else:
                    self.log(f"❌ Streaming failed: Only {len(chunks)} chunks received, expected at least {min_chunks_required}")
            
            self.log(f"   Response: {full_response}")
            
            return {
                "test": "streaming",
                "success": success,
                "duration": duration,
                "chunks": len(chunks),
                "response": full_response,
                "has_all_numbers": has_all_numbers,
                "min_chunks_required": min_chunks_required
            }
            
        except Exception as e:
            import traceback
            self.log(f"❌ Streaming failed with exception: {str(e)}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return {
                "test": "streaming",
                "success": False,
                "error": str(e)
            }
    
    def test_streaming_raw_http(self) -> Dict[str, Any]:
        """Test streaming response using raw HTTP to see what proxy returns"""
        self.log("Testing streaming response with raw HTTP...")
        
        import httpx
        
        start_time = time.time()
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": "Count from 1 to 5, one number at a time."}
                    ],
                    "stream": True,
                    "extra_body": {"cache": {"no-cache": True}}
                },
                headers={"Authorization": "Bearer dummy-key"},
                timeout=30.0
            ) as response:
                self.log(f"HTTP Response status: {response.status_code}")
                self.log(f"HTTP Response headers: {dict(response.headers)}")
                
                lines = []
                for line in response.iter_lines():
                    self.log(f"Raw line: '{line}'")
                    lines.append(line)
                    if len(lines) >= 10:  # Limit to prevent too much output
                        break
                
                self.log(f"Total lines received: {len(lines)}")
                
                return {
                    "test": "streaming_raw_http",
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "lines_count": len(lines),
                    "first_few_lines": lines[:5]
                }
                
        except Exception as e:
            import traceback
            self.log(f"❌ Raw HTTP streaming failed: {str(e)}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return {
                "test": "streaming_raw_http",
                "success": False,
                "error": str(e)
            }
    
    def test_caching(self) -> Dict[str, Any]:
        """Test caching behavior"""
        self.log("Testing caching behavior...")
        
        # Use deterministic request for caching
        messages = [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ]
        
        results = []
        
        # First request (should miss cache)
        self.log("  Making first request (expecting cache miss)...")
        start1 = time.time()
        try:
            response1 = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            duration1 = time.time() - start1
            content1 = response1.choices[0].message.content
            
            self.log(f"  First request: {content1} (took {duration1:.2f}s)")
            results.append({"request": 1, "duration": duration1, "content": content1})
            
        except Exception as e:
            self.log(f"❌ First request failed: {str(e)}", "ERROR")
            return {"test": "caching", "success": False, "error": str(e)}
        
        # Second request (should hit cache)
        self.log("  Making second request (expecting cache hit)...")
        start2 = time.time()
        try:
            response2 = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            duration2 = time.time() - start2
            content2 = response2.choices[0].message.content
            
            self.log(f"  Second request: {content2} (took {duration2:.2f}s)")
            results.append({"request": 2, "duration": duration2, "content": content2})
            
            # Cache hit should be much faster
            speedup = duration1 / duration2 if duration2 > 0 else float('inf')
            self.log(f"✅ Caching test complete: {speedup:.1f}x speedup")
            
            return {
                "test": "caching",
                "success": True,
                "results": results,
                "speedup": speedup,
                "cache_worked": duration2 < duration1 * 0.5  # At least 2x faster
            }
            
        except Exception as e:
            self.log(f"❌ Second request failed: {str(e)}", "ERROR")
            return {"test": "caching", "success": False, "error": str(e)}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid request"""
        self.log("Testing error handling...")
        
        try:
            response = self.client.chat.completions.create(
                model="invalid-model-name",
                messages=[{"role": "user", "content": "Test"}],
            )
            
            self.log("❌ Expected error but got success", "ERROR")
            return {
                "test": "error_handling",
                "success": False,
                "error": "Expected error for invalid model"
            }
            
        except Exception as e:
            error_msg = str(e)
            self.log(f"✅ Got expected error: {error_msg}")
            return {
                "test": "error_handling",
                "success": True,
                "error_type": type(e).__name__,
                "error_message": error_msg
            }
    
    def test_streaming_caching(self) -> Dict[str, Any]:
        """Test caching behavior for streaming requests"""
        self.log("Testing streaming caching behavior...")
        
        # Use deterministic request for caching
        messages = [
            {"role": "user", "content": "List exactly three colors, one per line."}
        ]
        
        results = []
        
        # First streaming request (should miss cache)
        self.log("  Making first streaming request (expecting cache miss)...")
        start1 = time.time()
        try:
            stream1 = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                # Enable streaming cache
                extra_body={"cache": {"stream-cache": True}},
            )
            
            chunks1 = []
            for chunk in stream1:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content:
                        chunks1.append(chunk.choices[0].delta.content)
            
            duration1 = time.time() - start1
            content1 = "".join(chunks1)
            
            self.log(f"  First request: {content1.strip()} (took {duration1:.2f}s)")
            results.append({"request": 1, "duration": duration1, "content": content1, "chunks": len(chunks1)})
            
        except Exception as e:
            self.log(f"❌ First streaming request failed: {str(e)}", "ERROR")
            return {"test": "streaming_caching", "success": False, "error": str(e)}
        
        # Small delay to ensure cache is written
        time.sleep(0.5)
        
        # Second streaming request (should hit cache)
        self.log("  Making second streaming request (expecting cache hit)...")
        start2 = time.time()
        try:
            stream2 = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                # Enable streaming cache
                extra_body={"cache": {"stream-cache": True}},
            )
            
            chunks2 = []
            for chunk in stream2:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content:
                        chunks2.append(chunk.choices[0].delta.content)
            
            duration2 = time.time() - start2
            content2 = "".join(chunks2)
            
            self.log(f"  Second request: {content2.strip()} (took {duration2:.2f}s)")
            results.append({"request": 2, "duration": duration2, "content": content2, "chunks": len(chunks2)})
            
            # Check if content is the same (cached properly)
            content_matches = content1 == content2
            
            # Cache hit should be much faster
            speedup = duration1 / duration2 if duration2 > 0 else float('inf')
            cache_worked = duration2 < duration1 * 0.5  # At least 2x faster
            
            self.log(f"✅ Streaming caching test complete: {speedup:.1f}x speedup, content matches: {content_matches}")
            
            return {
                "test": "streaming_caching",
                "success": content_matches and cache_worked,
                "results": results,
                "speedup": speedup,
                "cache_worked": cache_worked,
                "content_matches": content_matches
            }
            
        except Exception as e:
            self.log(f"❌ Second streaming request failed: {str(e)}", "ERROR")
            return {"test": "streaming_caching", "success": False, "error": str(e), "results": results}
    
    def test_responses_api_basic(self) -> Dict[str, Any]:
        """Test basic responses API (non-streaming)"""
        self.log("Testing responses API basic...")
        
        start_time = time.time()
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions="You are a helpful assistant. Be very concise.",
                input="What is 2+2? Reply with just the number and nothing else.",
                extra_body={"cache": {"no-cache": True}},
            )
            
            duration = time.time() - start_time
            output = response.output_text
            
            self.log(f"✅ Responses API success: {output} (took {duration:.2f}s)")
            
            return {
                "test": "responses_api_basic",
                "success": True,
                "duration": duration,
                "output": output
            }
            
        except Exception as e:
            self.log(f"❌ Responses API failed: {str(e)}", "ERROR")
            return {
                "test": "responses_api_basic",
                "success": False,
                "error": str(e)
            }
    
    def test_responses_api_streaming(self) -> Dict[str, Any]:
        """Test responses API streaming"""
        self.log("Testing responses API streaming...")
        
        start_time = time.time()
        try:
            self.log("Creating responses API streaming request...")
            stream = self.client.responses.create(
                model=self.model,
                instructions="You are a helpful assistant.",
                input="Count from 1 to 5, one number at a time.",
                stream=True,
                extra_body={"cache": {"no-cache": True}},
            )
                        
            chunks = []
            chunk_count = 0
            for event in stream:
                chunk_count += 1
                if hasattr(event, 'type') and event.type == "response.output_text.delta":
                    if hasattr(event, 'delta') and event.delta:
                        chunks.append(event.delta)
            
            self.log(f"Responses API stream iteration completed. Total events processed: {chunk_count}")
            
            duration = time.time() - start_time
            full_response = "".join(chunks)
            
            # Check if we received the expected content (numbers 1-5)
            has_all_numbers = all(str(i) in full_response for i in range(1, 6))
            
            # We should have received at least one chunk with content
            min_chunks_required = 1
            success = len(chunks) >= min_chunks_required and has_all_numbers
            
            if success:
                self.log(f"✅ Responses API streaming success: {len(chunks)} chunks received with all numbers 1-5 (took {duration:.2f}s)")
            else:
                if not has_all_numbers:
                    missing_numbers = [str(i) for i in range(1, 6) if str(i) not in full_response]
                    self.log(f"❌ Responses API streaming failed: Missing numbers {missing_numbers} in response")
                else:
                    self.log(f"❌ Responses API streaming failed: Only {len(chunks)} chunks received, expected at least {min_chunks_required}")
            
            self.log(f"   Response: {full_response}")
            
            return {
                "test": "responses_api_streaming",
                "success": success,
                "duration": duration,
                "chunks": len(chunks),
                "response": full_response,
                "has_all_numbers": has_all_numbers,
                "min_chunks_required": min_chunks_required
            }
            
        except Exception as e:
            import traceback
            self.log(f"❌ Responses API streaming failed with exception: {str(e)}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return {
                "test": "responses_api_streaming",
                "success": False,
                "error": str(e)
            }
    
    def test_load_balancing(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test load balancing by making multiple requests"""
        self.log(f"Testing load balancing with {num_requests} requests...")
        
        request_times = []
        
        for i in range(num_requests):
            start = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f"Say 'Request {i+1}' and nothing else."}
                    ],
                    extra_body={"cache": {"no-cache": True}},
                )
                duration = time.time() - start
                request_times.append(duration)
                
                self.log(f"  Request {i+1}/{num_requests}: {duration:.2f}s")
                
            except Exception as e:
                self.log(f"  Request {i+1} failed: {str(e)}", "ERROR")
        
        if request_times:
            avg_time = sum(request_times) / len(request_times)
            self.log(f"✅ Load balancing test complete: {len(request_times)}/{num_requests} succeeded")
            self.log(f"   Average response time: {avg_time:.2f}s")
            
            return {
                "test": "load_balancing",
                "success": True,
                "total_requests": num_requests,
                "successful_requests": len(request_times),
                "average_time": avg_time,
                "request_times": request_times
            }
        else:
            return {
                "test": "load_balancing",
                "success": False,
                "error": "All requests failed"
            }
    
    async def test_concurrent_requests(self, num_concurrent: int = 5) -> Dict[str, Any]:
        """Test concurrent request handling"""
        self.log(f"Testing {num_concurrent} concurrent requests...")
        
        async def make_request(index: int) -> Dict[str, Any]:
            """Make a single async request"""
            start = time.time()
            try:
                # Use httpx for async requests
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "user", "content": f"Say 'Concurrent {index}' and nothing else."}
                            ],
                            "extra_body": {"cache": {"no-cache": True}}
                        },
                        headers={"Authorization": "Bearer dummy-key"}
                    )
                    
                    duration = time.time() - start
                    if response.status_code == 200:
                        data = response.json()
                        content = data['choices'][0]['message']['content']
                        return {"index": index, "success": True, "duration": duration, "content": content}
                    else:
                        return {"index": index, "success": False, "duration": duration, "error": response.text}
                        
            except Exception as e:
                duration = time.time() - start
                return {"index": index, "success": False, "duration": duration, "error": str(e)}
        
        # Run concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time
        
        successful = [r for r in results if r['success']]
        self.log(f"✅ Concurrent test complete: {len(successful)}/{num_concurrent} succeeded in {total_duration:.2f}s")
        
        return {
            "test": "concurrent_requests",
            "success": len(successful) > 0,
            "num_concurrent": num_concurrent,
            "successful": len(successful),
            "total_duration": total_duration,
            "results": results
        }
    
    async def check_health_and_stats(self) -> Dict[str, Any]:
        """Check health and stats endpoints"""
        self.log("Checking health and stats endpoints...")
        
        async with httpx.AsyncClient() as client:
            # Check health
            health_resp = await client.get(f"{self.base_url}/health")
            health_data = health_resp.json() if health_resp.status_code == 200 else None
            
            # Check stats
            stats_resp = await client.get(f"{self.base_url}/stats")
            stats_data = stats_resp.json() if stats_resp.status_code == 200 else None
            
            if health_data:
                self.log(f"✅ Health check: {health_data.get('status', 'unknown')}")
                self.log(f"   Redis: {health_data.get('redis', 'unknown')}")
                self.log(f"   Model groups: {health_data.get('model_groups', [])}")
                
            if stats_data:
                self.log("✅ Stats retrieved successfully")
                # Show endpoint stats
                for model_group, endpoints in stats_data.get('endpoints', {}).items():
                    self.log(f"   {model_group}:")
                    for ep in endpoints:
                        self.log(f"     - {ep['base_url']}: {ep['total_requests']} requests, "
                               f"{ep['success_rate']:.1f}% success rate")
            
            return {
                "test": "health_stats",
                "success": health_data is not None and stats_data is not None,
                "health": health_data,
                "stats": stats_data
            }
    
    async def test_responses_api_async(self) -> Dict[str, Any]:
        """Test responses API with AsyncOpenAI client (like the simple_streaming_test.py example)"""
        self.log("Testing responses API with AsyncOpenAI...")
        
        from openai import AsyncOpenAI
        
        async_client = AsyncOpenAI(base_url=self.base_url)
        
        start_time = time.time()
        try:
            # Test non-streaming first
            self.log("  Testing async non-streaming...")
            response = await async_client.responses.create(
                model=self.model,
                instructions="You are a helpful assistant. Be very concise.",
                input="What is the capital of France? Reply with just the city name.",
            )
            
            output = response.output_text
            self.log(f"  Non-streaming result: {output}")
            
            # Test streaming
            self.log("  Testing async streaming...")
            stream = await async_client.responses.create(
                model=self.model,
                input="Write a one-sentence bedtime story about a unicorn.",
                stream=True,
            )
            
            chunks = []
            async for event in stream:
                if event.type == "response.output_text.delta":
                    chunks.append(event.delta)
            
            streaming_output = "".join(chunks)
            duration = time.time() - start_time
            
            self.log(f"✅ Async responses API success (took {duration:.2f}s)")
            self.log(f"  Streaming result: {streaming_output}")
            
            return {
                "test": "responses_api_async",
                "success": True,
                "duration": duration,
                "non_streaming_output": output,
                "streaming_output": streaming_output,
                "streaming_chunks": len(chunks)
            }
            
        except Exception as e:
            import traceback
            self.log(f"❌ Async responses API failed: {str(e)}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return {
                "test": "responses_api_async",
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all tests"""
        self.log("=== Starting LLMProxy Live Tests ===\n")
        
        tests = [
            self.test_basic_completion,
            self.test_streaming,
            self.test_streaming_raw_http,
            self.test_caching,
            self.test_error_handling,
            self.test_streaming_caching,
            self.test_responses_api_basic,
            self.test_responses_api_streaming,
            lambda: self.test_load_balancing(5),  # Fewer requests for quick testing
        ]
        
        for test_func in tests:
            result = test_func()
            self.results.append(result)
            print()  # Blank line between tests
        
        # Run async tests
        async def run_async_tests():
            concurrent_result = await self.test_concurrent_requests(3)
            self.results.append(concurrent_result)
            print()
            
            health_stats_result = await self.check_health_and_stats()
            self.results.append(health_stats_result)
            
            responses_api_async_result = await self.test_responses_api_async()
            self.results.append(responses_api_async_result)
        
        asyncio.run(run_async_tests())
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print("=== Test Summary ===")
        print("="*50)
        
        passed = sum(1 for r in self.results if r.get('success', False))
        total = len(self.results)
        
        print(f"\nTotal: {passed}/{total} tests passed")
        print("\nResults by test:")
        
        for result in self.results:
            test_name = result.get('test', 'unknown')
            success = result.get('success', False)
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name:.<30} {status}")
            
            # Show additional details for specific tests
            if test_name == "caching" and success:
                speedup = result.get('speedup', 0)
                cache_worked = result.get('cache_worked', False)
                print(f"    └─ Speedup: {speedup:.1f}x, Cache: {'Working' if cache_worked else 'Not working'}")
            elif test_name == "streaming_caching" and success:
                speedup = result.get('speedup', 0)
                cache_worked = result.get('cache_worked', False)
                content_matches = result.get('content_matches', False)
                print(f"    └─ Speedup: {speedup:.1f}x, Cache: {'Working' if cache_worked else 'Not working'}, Content matches: {content_matches}")
            elif test_name == "load_balancing" and success:
                avg_time = result.get('average_time', 0)
                print(f"    └─ Avg response time: {avg_time:.2f}s")
            elif test_name == "concurrent_requests" and success:
                successful = result.get('successful', 0)
                total_concurrent = result.get('num_concurrent', 0)
                print(f"    └─ Successful: {successful}/{total_concurrent}")
            elif test_name == "responses_api_streaming" and success:
                chunks = result.get('chunks', 0)
                print(f"    └─ Streaming chunks: {chunks}")
            elif test_name == "responses_api_async" and success:
                chunks = result.get('streaming_chunks', 0)
                print(f"    └─ Async streaming chunks: {chunks}")
        
        print("\n" + "="*50)


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLMProxy with live requests")
    parser.add_argument("--proxy-url", default=None,
                       help="Proxy URL (default: read from config)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Get proxy URL (from args or config)
    proxy_url = args.proxy_url if args.proxy_url else get_proxy_url()
    
    # Check if proxy is running
    try:
        print(f"Checking proxy health at {proxy_url}/health")
        response = httpx.get(f"{proxy_url}/health", timeout=2)
        if response.status_code != 200:
            print(f"❌ Proxy health check failed: {response.status_code}")
            print("Make sure LLMProxy is running!")
            return
    except Exception as e:
        print(f"❌ Cannot connect to proxy at {proxy_url}")
        print(f"Error: {e}")
        print("\nMake sure LLMProxy is running:")
        print("  python run_proxy.py")
        return
    
    # Run tests
    tester = ProxyTester(proxy_url)
    
    if args.quick:
        print("Running quick tests...\n")
        result = tester.test_basic_completion()
        tester.results.append(result)
        
        # Also test streaming specifically
        streaming_result = tester.test_streaming()
        tester.results.append(streaming_result)
        
        # Test raw HTTP streaming
        raw_streaming_result = tester.test_streaming_raw_http()
        tester.results.append(raw_streaming_result)
        
        # Test responses API
        responses_result = tester.test_responses_api_basic()
        tester.results.append(responses_result)
        
        responses_streaming_result = tester.test_responses_api_streaming()
        tester.results.append(responses_streaming_result)
        
        tester.print_summary()
    else:
        tester.run_all_tests()


if __name__ == "__main__":
    main() 