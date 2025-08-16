import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
openai_client = AsyncOpenAI()
proxy_client = AsyncOpenAI(base_url="http://localhost:4243/")


async def main():
    # Same prompt for both requests
    prompt = "Write a one-sentence bedtime story about a unicorn. Come up with a very creative story."

    # First request
    print("Making first request...")
    start_time = time.time()
    response1 = await proxy_client.responses.create(
        model="gpt-4.1",
        input=prompt,
        temperature=0.0,
    )
    end_time = time.time()
    first_request_time = end_time - start_time

    print(f"First request completed in {first_request_time:.3f} seconds")
    print(f"Response 1: {response1.output_text}")
    print()

    # Second request (identical)
    print("Making second request...")
    start_time = time.time()
    response2 = await proxy_client.responses.create(
        model="gpt-4.1",
        input=prompt,
        temperature=0.0,
    )
    end_time = time.time()
    second_request_time = end_time - start_time

    print(f"Second request completed in {second_request_time:.3f} seconds")
    print(f"Response 2: {response2.output_text}")
    print()

    # Summary
    print("=== TIMING SUMMARY ===")
    print(f"First request:  {first_request_time:.3f} seconds")
    print(f"Second request: {second_request_time:.3f} seconds")
    print(
        f"Time difference: {abs(first_request_time - second_request_time):.3f} seconds"
    )

    if second_request_time < first_request_time:
        speedup = first_request_time / second_request_time
        if speedup >= 10:
            print(f"Second request was {speedup:.2f}x faster - CACHED!")
        else:
            print(
                f"Second request was {speedup:.2f}x faster - but not enough for caching (need 10x+)"
            )
    else:
        print("Second request was not faster - NOT CACHED")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
