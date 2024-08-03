# async_llm_handler/utils/rate_limiter.py

import asyncio
import time

class RateLimitTimeoutError(Exception):
    pass

class RateLimiter:
    def __init__(self, rate: int, period: int = 60, timeout: float = 30.0):
        self.rate = rate
        self.period = period
        self.allowance = rate
        self.last_check = time.monotonic()
        self.timeout = timeout
        self._lock = asyncio.Lock()

    async def acquire_async(self):
        async with self._lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.period)
            if self.allowance > self.rate:
                self.allowance = self.rate
            if self.allowance < 1:
                wait_time = (1 - self.allowance) / (self.rate / self.period)
                try:
                    await asyncio.wait_for(asyncio.sleep(wait_time), timeout=self.timeout)
                except asyncio.TimeoutError:
                    raise RateLimitTimeoutError(f"Rate limit wait exceeded timeout of {self.timeout} seconds")
            else:
                self.allowance -= 1

    def release(self):
        pass  # No action needed for release in this implementation