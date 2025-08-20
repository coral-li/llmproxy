import time

import pytest
from openai import AsyncOpenAI


@pytest.mark.asyncio
async def test_responses_api_two_identical_requests_cache_speedup(
    async_openai_client: AsyncOpenAI, model: str
):
    """Replicate a simple script that makes two identical Responses API requests using AsyncOpenAI.

    Expectation: the second request should be significantly faster if caching works.
    No artificial delay is used; this surfaces potential cache misses in realistic settings.
    """

    prompt = (
        "Write a one-sentence bedtime story about a unicorn. "
        "Come up with a very creative story."
    )

    # First request
    start1 = time.time()
    resp1 = await async_openai_client.responses.create(
        model=model,
        input=prompt,
        temperature=0.0,
    )
    duration1 = time.time() - start1
    assert getattr(resp1, "output_text", None), "First response has no output_text"

    # Second (identical) request
    start2 = time.time()
    resp2 = await async_openai_client.responses.create(
        model=model,
        input=prompt,
        temperature=0.0,
    )
    duration2 = time.time() - start2
    assert getattr(resp2, "output_text", None), "Second response has no output_text"

    # The second request should be much faster if caching works properly (>=10x)
    assert (
        duration2 * 10 < duration1
    ), f"Second request not >=10x faster: {duration1:.3f}s vs {duration2:.3f}s"
