# File: async_llm_handler/utils/rate_limiter.py

import asyncio
import time

class RateLimiter:
    def __init__(self, rate: int, period: int = 60):
        self.rate = rate
        self.period = period
        self.allowance = rate
        self.last_check = time.monotonic()

    async def __aenter__(self):
        while True:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.period)
            if self.allowance > self.rate:
                self.allowance = self.rate
            if self.allowance < 1:
                await asyncio.sleep(1)
                continue
            self.allowance -= 1
            break
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass