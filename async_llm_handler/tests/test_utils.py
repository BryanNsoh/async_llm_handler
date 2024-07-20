# File: async_llm_handler/tests/test_utils.py

import pytest
from async_llm_handler.utils import count_tokens, clip_prompt, RateLimiter

def test_count_tokens():
    text = "Hello, world!"
    assert count_tokens(text) > 0

def test_clip_prompt():
    long_prompt = "This is a very long prompt " * 100
    max_tokens = 10
    clipped = clip_prompt(long_prompt, max_tokens)
    assert count_tokens(clipped) <= max_tokens

@pytest.mark.asyncio
async def test_rate_limiter():
    limiter = RateLimiter(rate=2, period=1)
    
    start_time = pytest.helpers.time()
    
    async with limiter:
        pass
    async with limiter:
        pass
    
    # This should wait
    async with limiter:
        pass
    
    end_time = pytest.helpers.time()
    
    assert end_time - start_time >= 1.0

def test_logger():
    from async_llm_handler.utils import get_logger
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == 20  # INFO level