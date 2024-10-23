# tests/test_llm_handler.py

import os
import json
import asyncio
import unittest
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, AsyncMock, MagicMock
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from async_llm_handler import (
    LLMAPIHandler,
    BatchResult,
    LLMResponse,
    RetrySettings,
    ModelLimits,
    TokenEncoder
)
from test_models import (
    StructuredOutput,
    BatchResult as TestBatchResult,
    CompletionMetadata
)

class TestLLMAPIHandler(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Allow custom rate limits for testing
        custom_rate_limits = {
            'gpt-4o-mini': {
                'rpm': 100,  # Reduced rate for testing
                'tpm': 5000,
                'max_tokens': 16000,
                'context_window': 128000
            },
            'claude-3-5-sonnet-20241022': {
                'rpm': 100,
                'tpm': 5000,
                'max_tokens': 8000,
                'context_window': 200000
            }
        }
        self.handler = LLMAPIHandler(custom_rate_limits=custom_rate_limits)
        self.test_prompts = {
            "simple": "Say hello",
            "structured": "Generate a greeting",
            "malformed_json": "Generate malformed JSON",
            "empty_prompt": "",
            "unsupported_model": "This should fail"
        }
        self.models = {
            "openai": "gpt-4o-mini",
            "claude": "claude-3-5-sonnet-20241022",
            "unsupported": "invalid-model"
        }

    async def asyncTearDown(self):
        if hasattr(self, 'handler'):
            await self.handler.reset_metrics()
            if hasattr(self.handler, 'async_openai_client'):
                await self.handler.async_openai_client.close()
            if hasattr(self.handler, 'async_anthropic_client'):
                await self.handler.async_anthropic_client.close()

    async def test_openai_basic_completion(self):
        """Test basic OpenAI completion with minimal token usage"""
        response = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"]
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    async def test_claude_basic_completion(self):
        """Test basic Claude completion with minimal token usage"""
        response = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["claude"]
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    async def test_structured_output_openai(self):
        """Test OpenAI structured output with Pydantic model"""
        system_msg = "Return response as valid JSON with text and metadata fields"
        response = await self.handler.process(
            self.test_prompts["structured"],
            model=self.models["openai"],
            system_message=system_msg,
            response_format=StructuredOutput
        )
        self.assertIsInstance(response, StructuredOutput)
        self.assertIsInstance(response.metadata, CompletionMetadata)
        self.assertGreater(len(response.text), 0)

    async def test_structured_output_async_batch_openai(self):
        """Test OpenAI structured output in async batch mode with Pydantic model"""
        system_msg = "Return response as valid JSON with text and metadata fields"
        prompts = [self.test_prompts["structured"]] * 2  # Minimal token usage
        result = await self.handler.process(
            prompts,
            model=self.models["openai"],
            mode="async_batch",
            system_message=system_msg,
            response_format=StructuredOutput
        )
        self.assertIsInstance(result, BatchResult)
        self.assertEqual(result.metadata['total_prompts'], 2)
        self.assertEqual(len(result.results), 2)
        for res in result.results:
            self.assertIn('prompt', res)
            self.assertIn('response', res)
            self.assertIsInstance(res['response'], StructuredOutput)

    async def test_structured_output_openai_batch(self):
        """Test OpenAI structured output in OpenAI Batch API mode with Pydantic model"""
        system_msg = "Return response as valid JSON with text and metadata fields"
        prompts = [self.test_prompts["structured"]] * 2  # Minimal token usage
        result = await self.handler.process(
            prompts,
            model=self.models["openai"],
            mode="openai_batch",
            system_message=system_msg,
            response_format=StructuredOutput,
            output_dir="test_output"
        )
        self.assertIsInstance(result, BatchResult)
        self.assertEqual(result.metadata['num_requests'], 2)
        self.assertEqual(len(result.results), 2)
        for res in result.results:
            self.assertIn('prompt', res)
            self.assertIn('response', res)
            self.assertIsInstance(res['response'], StructuredOutput)

    async def test_batch_processing_empty_prompts(self):
        """Test batch processing with empty prompts list"""
        prompts = []
        with self.assertRaises(ValueError) as context:
            await self.handler.process(
                prompts,
                model=self.models["openai"],
                mode="async_batch"
            )
        self.assertIn("Prompt list is empty", str(context.exception))
        metrics = await self.handler.get_metrics()
        self.assertEqual(metrics['failed_requests'], 1)

    async def test_batch_processing_deduplicate_prompts(self):
        """Test batch processing with duplicate prompts and deduplication"""
        prompts = [self.test_prompts["simple"]] * 5  # Duplicate prompts
        result = await self.handler.process(
            prompts,
            model=self.models["openai"],
            mode="async_batch",
            deduplicate_prompts=True
        )
        self.assertIsInstance(result, BatchResult)
        self.assertEqual(result.metadata['total_prompts'], 1)
        self.assertEqual(len(result.results), 1)

    async def test_metrics_tracking(self):
        """Test metrics tracking after multiple requests"""
        await self.handler.reset_metrics()
        # Perform multiple requests
        for _ in range(3):
            await self.handler.process(
                self.test_prompts["simple"],
                model=self.models["openai"]
            )
        metrics = await self.handler.get_metrics()
        self.assertEqual(metrics['total_requests'], 3)
        self.assertEqual(metrics['successful_requests'], 3)
        self.assertEqual(metrics['failed_requests'], 0)
        self.assertGreater(metrics['total_tokens'], 0)
        self.assertGreater(metrics['total_processing_time'], 0.0)
        self.assertGreater(metrics['success_rate'], 0.0)

    def test_token_encoder(self):
        """Test token encoding with minimal token usage"""
        text = "Test message"
        for model in self.models.values():
            if model == "invalid-model":
                with self.assertRaises(ValueError):
                    TokenEncoder.estimate_tokens(text, model)
            else:
                count = TokenEncoder.estimate_tokens(text, model)
                self.assertIsInstance(count, int)
                self.assertGreater(count, 0)

    async def test_error_handling_with_max_retries(self):
        """Test error handling with mocked API errors to avoid token usage"""
        test_error = Exception("Test API Error")
        await self.handler.reset_metrics()

        with patch.object(
            self.handler.async_openai_client.chat.completions,
            'create',
            new_callable=AsyncMock,
            side_effect=test_error
        ) as mock_create:
            with self.assertRaises(Exception) as context:
                await self.handler.process(
                    self.test_prompts["simple"],
                    model=self.models["openai"]
                )
            self.assertEqual(mock_create.call_count, RetrySettings.MAX_RETRIES)
            self.assertEqual(str(context.exception), "Test API Error")
            metrics = await self.handler.get_metrics()
            self.assertEqual(metrics['failed_requests'], 1)  # Should increment only once
            self.assertEqual(metrics['successful_requests'], 0)

    async def test_error_handling_unsupported_model(self):
        """Test handling of unsupported model names without API call"""
        with self.assertRaises(ValueError) as context:
            await self.handler.process(
                self.test_prompts["unsupported_model"],
                model=self.models["unsupported"]
            )
        self.assertIn("Unsupported model", str(context.exception))
        metrics = await self.handler.get_metrics()
        self.assertEqual(metrics['failed_requests'], 1)

    async def test_system_message_handling(self):
        """Test system message handling with minimal token usage"""
        system_msg = "You are a test assistant"
        response = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"],
            system_message=system_msg
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    async def test_temperature_control(self):
        """Test temperature parameter effect with minimal token usage"""
        response_low_temp = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"],
            temperature=0
        )
        response_high_temp = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"],
            temperature=1
        )
        self.assertIsInstance(response_low_temp, str)
        self.assertIsInstance(response_high_temp, str)
        self.assertGreater(len(response_low_temp), 0)
        self.assertGreater(len(response_high_temp), 0)
        # Optionally, compare responses to ensure variability
        # This is non-deterministic and may not always hold

    async def test_rate_limiting_behavior(self):
        """Test rate limiting behavior with mocked rate limit breach"""
        with patch.object(
            self.handler.request_limiters[self.models["openai"]],
            'acquire',
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError("Rate limit exceeded")
        ) as mock_acquire:
            with self.assertRaises(asyncio.TimeoutError):
                await self.handler.process(
                    self.test_prompts["simple"],
                    model=self.models["openai"]
                )
            self.assertTrue(mock_acquire.called)
            metrics = await self.handler.get_metrics()
            self.assertEqual(metrics['failed_requests'], 1)
            self.assertEqual(metrics['successful_requests'], 0)

    async def test_timeout_handling_with_mock(self):
        """Test handling of API timeouts with mocked delay"""
        with patch.object(
            self.handler.async_openai_client.chat.completions,
            'create',
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError("Request timed out")
        ) as mock_create:
            with self.assertRaises(asyncio.TimeoutError):
                await self.handler.process(
                    self.test_prompts["simple"],
                    model=self.models["openai"]
                )
            self.assertEqual(mock_create.call_count, RetrySettings.MAX_RETRIES)
            metrics = await self.handler.get_metrics()
            self.assertEqual(metrics['failed_requests'], 1)
            self.assertEqual(metrics['successful_requests'], 0)

    async def test_client_cleanup(self):
        """Test that clients are properly closed"""
        await self.handler.async_openai_client.close()
        await self.handler.async_anthropic_client.close()
        # Instead of checking 'closed', attempt to use the client and expect an exception
        with self.assertRaises(Exception):  # Replace with specific exception if known
            await self.handler.async_openai_client.chat.completions.create(
                model=self.models["openai"],
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.7
            )
        with self.assertRaises(Exception):  # Replace with specific exception if known
            await self.handler.async_anthropic_client.messages.create(
                model=self.models["claude"],
                messages=[{"role": "user", "content": "Test"}]
            )

if __name__ == '__main__':
    unittest.main()
