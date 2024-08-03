# async_llm_handler/handler.py

import asyncio
import json
import logging
from typing import Optional, Any, Dict
import aiohttp
import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI

from .config import Config
from .exceptions import LLMAPIError, RateLimitTimeoutError
from .utils.rate_limiter import RateLimiter
from .utils.token_utils import clip_prompt

logger = logging.getLogger(__name__)

class Handler:
    def __init__(self, config: Optional[Config] = None, rate_limit_timeout: float = 30.0, retry_attempts: int = 3, retry_delay: float = 1.0):
        self.config = config or Config()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._setup_clients()
        self._setup_rate_limiters(rate_limit_timeout)

    def _setup_clients(self):
        # Gemini setup
        genai.configure(api_key=self.config.gemini_api_key)
        self.gemini_client = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
        
        # Other clients setup
        self.claude_client = anthropic.AsyncAnthropic(api_key=self.config.claude_api_key)
        self.openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)

    def _setup_rate_limiters(self, timeout: float):
        self.rate_limiters = {
            'gemini_flash': RateLimiter(30, 60, timeout),
            'claude_3_5_sonnet': RateLimiter(5, 60, timeout),
            'claude_3_haiku': RateLimiter(5, 60, timeout),
            'gpt_4o': RateLimiter(5, 60, timeout),
            'gpt_4o_mini': RateLimiter(5, 60, timeout)
        }

    async def query(
        self,
        prompt: str,
        model: str,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        for attempt in range(self.retry_attempts):
            try:
                return await self._async_query(prompt, model, max_input_tokens, max_output_tokens, json_mode)
            except RateLimitTimeoutError as e:
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

    async def _async_query(self, prompt: str, model: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        method = getattr(self, f'_query_{model}_async', None)
        if not method:
            raise ValueError(f"Unsupported model for async query: {model}")
        
        return await method(prompt, max_input_tokens, max_output_tokens, json_mode)

    async def _query_gemini_flash_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        await self.rate_limiters['gemini_flash'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            logger.info("Generating content with Gemini Flash API (Async).")
            generation_config = {"response_mime_type": "application/json"} if json_mode else {}
            if max_output_tokens is not None:
                generation_config['max_output_tokens'] = max_output_tokens
            response = await self.gemini_client.generate_content_async(prompt, generation_config=generation_config)
            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Invalid response format from Gemini Flash API.")
        except Exception as e:
            logger.error(f"Error with Gemini Flash API: {e}")
            raise LLMAPIError(f"Gemini Flash API error: {str(e)}")
        finally:
            self.rate_limiters['gemini_flash'].release()

    async def _query_gpt_4o_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        await self.rate_limiters['gpt_4o'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            json_instruction = " Respond using JSON." if json_mode else ""
            messages = [{"role": "user", "content": prompt + json_instruction}]
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
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            response = await self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with GPT-4o API: {e}")
            raise LLMAPIError(f"GPT-4o API error: {str(e)}")
        finally:
            self.rate_limiters['gpt_4o'].release()

    async def _query_gpt_4o_mini_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        await self.rate_limiters['gpt_4o_mini'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            json_instruction = " Respond using JSON." if json_mode else ""
            messages = [{"role": "user", "content": prompt + json_instruction}]
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
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            response = await self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with GPT-4o mini API: {e}")
            raise LLMAPIError(f"GPT-4o mini API error: {str(e)}")
        finally:
            self.rate_limiters['gpt_4o_mini'].release()

    async def _query_claude_3_5_sonnet_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        await self.rate_limiters['claude_3_5_sonnet'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            json_instruction = "Respond using JSON." if json_mode else ""
            params = {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": prompt + json_instruction}],
                "max_tokens": max_output_tokens if max_output_tokens is not None else 4096,
            }
            response = await self.claude_client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude 3.5 Sonnet API: {e}")
            raise LLMAPIError(f"Claude 3.5 Sonnet API error: {str(e)}")
        finally:
            self.rate_limiters['claude_3_5_sonnet'].release()

    async def _query_claude_3_haiku_async(self, prompt: str, max_input_tokens: Optional[int] = None, max_output_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        await self.rate_limiters['claude_3_haiku'].acquire_async()
        try:
            if max_input_tokens:
                prompt = clip_prompt(prompt, max_input_tokens)
            json_instruction = "Respond using JSON." if json_mode else ""
            params = {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": prompt + json_instruction}],
                "max_tokens": max_output_tokens if max_output_tokens is not None else 4096,
            }
            response = await self.claude_client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude 3 Haiku API: {e}")
            raise LLMAPIError(f"Claude 3 Haiku API error: {str(e)}")
        finally:
            self.rate_limiters['claude_3_haiku'].release()