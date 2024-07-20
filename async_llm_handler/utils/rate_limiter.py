# File: async_llm_handler/utils/rate_limiter.py

import asyncio
import time

class RateLimiter:
    def __init__(self, rate: int, period: int = 60):
        self.rate = rate
        self.period = period
        self.allowance = rate
        self.last_check = time.monotonic()
        self._lock = asyncio.Lock()

    def acquire(self):
        current = time.monotonic()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.period)
        if self.allowance > self.rate:
            self.allowance = self.rate
        if self.allowance < 1:
            time.sleep((1 - self.allowance) / (self.rate / self.period))
            self.allowance = 0
        else:
            self.allowance -= 1

    async def acquire_async(self):
        async with self._lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.period)
            if self.allowance > self.rate:
                self.allowance = self.rate
            if self.allowance < 1:
                await asyncio.sleep((1 - self.allowance) / (self.rate / self.period))
                self.allowance = 0
            else:
                self.allowance -= 1

    def release(self):
        pass  # No action needed for release in this implementation