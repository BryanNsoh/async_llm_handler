# File: async_llm_handler/handler.py

import asyncio
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor
import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI

from .config import Config
from .exceptions import LLMAPIError
from .utils.rate_limiter import RateLimiter
from .utils.token_utils import clip_prompt
from .utils.logger import get_logger

logger = get_logger(__name__)

from typing import Union, Optional, Coroutine, Any

class LLMHandler:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._setup_clients()
        self._setup_rate_limiters()
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
        self.openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)

    def _setup_rate_limiters(self):
        self.rate_limiters = {
            'gemini_flash': RateLimiter(30, 60),
            'claude_3_5_sonnet': RateLimiter(5, 60),
            'claude_3_haiku': RateLimiter(5, 60),
            'gpt_4o': RateLimiter(5, 60),
            'gpt_4o_mini': RateLimiter(5, 60)
        }

    def query(self, prompt: str, model: str, sync: bool = True, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> Union[str, Coroutine[Any, Any, str]]:
        if sync:
            return self._sync_query(prompt, model, max_input_tokens, max_output_tokens)
        else:
            return self._async_query(prompt, model, max_input_tokens, max_output_tokens)

    def _sync_query(self, prompt: str, model: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        method = getattr(self, f'_query_{model}_sync', None)
        if not method:
            raise ValueError(f"Unsupported model for sync query: {model}")
        
        return method(prompt, max_input_tokens, max_output_tokens)

    async def _async_query(self, prompt: str, model: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        method = getattr(self, f'_query_{model}_async', None)
        if not method:
            raise ValueError(f"Unsupported model for async query: {model}")
        
        return await method(prompt, max_input_tokens, max_output_tokens)

    def _query_gemini_flash_sync(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        self.rate_limiters['gemini_flash'].acquire()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            logger.info("Generating content with Gemini Flash API (Sync).")
            params = {'max_output_tokens': max_output_tokens} if max_output_tokens is not None else {}
            response = self.gemini_client.generate_content(prompt, **params)
            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Invalid response format from Gemini Flash API.")
        except Exception as e:
            logger.error(f"Error with Gemini Flash API: {e}")
            raise LLMAPIError(f"Gemini Flash API error: {str(e)}")
        finally:
            self.rate_limiters['gemini_flash'].release()

    async def _query_gemini_flash_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        await self.rate_limiters['gemini_flash'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            logger.info("Generating content with Gemini Flash API (Async).")
            params = {'max_output_tokens': max_output_tokens} if max_output_tokens is not None else {}
            response = await self.gemini_client.generate_content_async(prompt, **params)
            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Invalid response format from Gemini Flash API.")
        except Exception as e:
            logger.error(f"Error with Gemini Flash API: {e}")
            raise LLMAPIError(f"Gemini Flash API error: {str(e)}")
        finally:
            self.rate_limiters['gemini_flash'].release()

    async def _query_gpt_4o_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        await self.rate_limiters['gpt_4o'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            messages = [{"role": "user", "content": prompt}]
            params = {
                "model": "gpt-4o-2024-05-13",
                "messages": messages,
                "temperature": 0.3,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            if max_output_tokens is not None:
                params["max_tokens"] = max_output_tokens
            response = await self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with GPT-4o API: {e}")
            raise LLMAPIError(f"GPT-4o API error: {str(e)}")
        finally:
            self.rate_limiters['gpt_4o'].release()

    def _query_gpt_4o_sync(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._query_gpt_4o_async(prompt, max_input_tokens, max_output_tokens))
        finally:
            loop.close()

    async def _query_gpt_4o_mini_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        await self.rate_limiters['gpt_4o_mini'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            messages = [{"role": "user", "content": prompt}]
            params = {
                "model": "gpt-4o-mini-2024-07-18",
                "messages": messages,
                "temperature": 0.3,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            if max_output_tokens is not None:
                params["max_tokens"] = max_output_tokens
            response = await self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with GPT-4o mini API: {e}")
            raise LLMAPIError(f"GPT-4o mini API error: {str(e)}")
        finally:
            self.rate_limiters['gpt_4o_mini'].release()

    def _query_gpt_4o_mini_sync(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._query_gpt_4o_mini_async(prompt, max_input_tokens, max_output_tokens))
        finally:
            loop.close()

    async def _query_claude_3_5_sonnet_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        await self.rate_limiters['claude_3_5_sonnet'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            params = {
                "model": "claude-3-5-sonnet-20240620",
                "messages": [{"role": "user", "content": prompt}],
                "system": "Directly fulfill the user's request without preamble, paying very close attention to all nuances of their instructions.",
                "max_tokens": max_output_tokens if max_output_tokens is not None else 4096,
            }
            response = await asyncio.to_thread(self.claude_client.messages.create, **params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude 3.5 Sonnet API: {e}")
            raise LLMAPIError(f"Claude 3.5 Sonnet API error: {str(e)}")
        finally:
            self.rate_limiters['claude_3_5_sonnet'].release()

    def _query_claude_3_5_sonnet_sync(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._query_claude_3_5_sonnet_async(prompt, max_input_tokens, max_output_tokens))
        finally:
            loop.close()

    async def _query_claude_3_haiku_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        await self.rate_limiters['claude_3_haiku'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            params = {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": prompt}],
                "system": "Directly fulfill the user's request without preamble, paying very close attention to all nuances of their instructions.",
                "max_tokens": max_output_tokens if max_output_tokens is not None else 4096,
            }
            response = await asyncio.to_thread(self.claude_client.messages.create, **params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude 3 Haiku API: {e}")
            raise LLMAPIError(f"Claude 3 Haiku API error: {str(e)}")
        finally:
            self.rate_limiters['claude_3_haiku'].release()

    def _query_claude_3_haiku_sync(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None) -> str:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._query_claude_3_haiku_async(prompt, max_input_tokens, max_output_tokens))
        finally:
            loop.close()