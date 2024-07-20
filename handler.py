# File: async_llm_handler/handler.py

import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import anthropic
import cohere
import google.generativeai as genai
from groq import Groq
from openai import AsyncOpenAI

from .config import Config
from .exceptions import LLMAPIError
from .utils.rate_limiter import RateLimiter
from .utils.token_utils import clip_prompt
from .utils.logger import get_logger

logger = get_logger(__name__)

class LLMHandler:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._setup_clients()
        self._setup_rate_limiters()
        self._loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
        self._executor = ThreadPoolExecutor()

    def _setup_clients(self):
        genai.configure(api_key=self.config.gemini_api_key)
        self.gemini_client = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            generation_config={"response_mime_type": "application/json"},
        )
        self.claude_client = anthropic.Anthropic(api_key=self.config.claude_api_key)
        self.cohere_client = cohere.Client(self.config.cohere_api_key)
        self.groq_client = Groq(api_key=self.config.groq_api_key)
        self.openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)

    def _setup_rate_limiters(self):
        self.rate_limiters = {
            'gemini': RateLimiter(30, 60),
            'claude': RateLimiter(5, 60),
            'openai': RateLimiter(5, 60),
            'cohere': RateLimiter(30, 60),
            'llama': RateLimiter(5, 60)
        }

    def query(self, prompt: str, model: str = 'auto') -> str:
        if self._loop and self._loop.is_running():
            return asyncio.run_coroutine_threadsafe(self._async_query(prompt, model), self._loop).result()
        else:
            return asyncio.run(self._async_query(prompt, model))

    async def _async_query(self, prompt: str, model: str = 'auto') -> str:
        if model == 'auto':
            return await self._query_all(prompt)
        
        method = getattr(self, f'_query_{model}', None)
        if not method:
            raise ValueError(f"Unsupported model: {model}")
        
        return await method(prompt)

    async def _query_all(self, prompt: str) -> str:
        methods = [
            self._query_gemini,
            self._query_cohere,
            self._query_llama,
            self._query_claude,
            self._query_openai
        ]
        
        for method in methods:
            try:
                return await method(prompt)
            except LLMAPIError:
                continue
        
        raise LLMAPIError("All LLM APIs failed to respond")

    async def _query_gemini(self, prompt: str) -> str:
        async with self.rate_limiters['gemini']:
            try:
                clipped_prompt = clip_prompt(prompt, max_tokens=500000)
                logger.info("Generating content with Gemini API.")
                response = await self.gemini_client.generate_content_async(clipped_prompt)
                if response.candidates:
                    return response.candidates[0].content.parts[0].text
                else:
                    raise ValueError("Invalid response format from Gemini API.")
            except Exception as e:
                logger.error(f"Error with Gemini API: {e}")
                raise LLMAPIError(f"Gemini API error: {str(e)}")

    async def _query_cohere(self, prompt: str) -> str:
        async with self.rate_limiters['cohere']:
            try:
                clipped_prompt = clip_prompt(prompt, max_tokens=100000)
                response = self.cohere_client.chat(message=clipped_prompt, model="command-r")
                return response.text
            except Exception as e:
                logger.error(f"Error with Cohere API: {e}")
                raise LLMAPIError(f"Cohere API error: {str(e)}")

    async def _query_openai(self, prompt: str) -> str:
        async with self.rate_limiters['openai']:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=None,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error with OpenAI API: {e}")
                raise LLMAPIError(f"OpenAI API error: {str(e)}")

    async def _query_claude(self, prompt: str) -> str:
        async with self.rate_limiters['claude']:
            try:
                clipped_prompt = clip_prompt(prompt, max_tokens=180000)
                response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=3000,
                    system="Directly fulfill the user's request without preamble, paying very close attention to all nuances of their instructions.",
                    messages=[{"role": "user", "content": clipped_prompt}],
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Error with Claude API: {e}")
                raise LLMAPIError(f"Claude API error: {str(e)}")

    async def _query_llama(self, prompt: str) -> str:
        async with self.rate_limiters['llama']:
            try:
                clipped_prompt = clip_prompt(prompt, max_tokens=8192)
                logger.info("Generating content with Groq API.")
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": clipped_prompt}],
                    model="llama3-8b-8192",
                )
                if response.choices:
                    return response.choices[0].message.content
                else:
                    raise ValueError("Invalid response format from Groq API.")
            except Exception as e:
                logger.error(f"Error with Groq API: {e}")
                raise LLMAPIError(f"Groq API error: {str(e)}")