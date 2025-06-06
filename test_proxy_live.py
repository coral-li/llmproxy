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
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Count from 1 to 5, one number at a time."}
                ],
                stream=True
            )
            
            chunks = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            
            duration = time.time() - start_time
            full_response = "".join(chunks)
            
            self.log(f"✅ Streaming success: {len(chunks)} chunks received (took {duration:.2f}s)")
            self.log(f"   Response: {full_response}")
            
            return {
                "test": "streaming",
                "success": True,
                "duration": duration,
                "chunks": len(chunks),
                "response": full_response
            }
            
        except Exception as e:
            self.log(f"❌ Streaming failed: {str(e)}", "ERROR")
            return {
                "test": "streaming",
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
                            ]                        
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
    
    def run_all_tests(self):
        """Run all tests"""
        self.log("=== Starting LLMProxy Live Tests ===\n")
        
        tests = [
            self.test_basic_completion,
            self.test_streaming,
            self.test_caching,
            self.test_error_handling,
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
            elif test_name == "load_balancing" and success:
                avg_time = result.get('average_time', 0)
                print(f"    └─ Avg response time: {avg_time:.2f}s")
            elif test_name == "concurrent_requests" and success:
                successful = result.get('successful', 0)
                total_concurrent = result.get('num_concurrent', 0)
                print(f"    └─ Successful: {successful}/{total_concurrent}")
        
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
        tester.print_summary()
    else:
        tester.run_all_tests()


if __name__ == "__main__":
    main() 