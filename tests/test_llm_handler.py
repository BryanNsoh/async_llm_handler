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
        self.handler = LLMAPIHandler()
        self.test_prompts = {
            "simple": "Say hello",
            "structured": "Generate a greeting"
        }
        self.models = {
            "openai": "gpt-4o-mini",
            "claude": "claude-3-5-sonnet-20241022"
        }

    async def asyncTearDown(self):
        if hasattr(self, 'handler'):
            await self.handler.reset_metrics()
            if hasattr(self.handler, 'async_openai_client'):
                await self.handler.async_openai_client.close()
            if hasattr(self.handler, 'async_anthropic_client'):
                await self.handler.async_anthropic_client.close()

    async def test_openai_basic_completion(self):
        """Test basic OpenAI completion"""
        response = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"]
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    async def test_claude_basic_completion(self):
        """Test basic Claude completion"""
        response = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["claude"]
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    async def test_structured_output_openai(self):
        """Test OpenAI structured output"""
        system_msg = "Return response as valid JSON with text and metadata fields"
        response = await self.handler.process(
            self.test_prompts["structured"],
            model=self.models["openai"],
            system_message=system_msg,
            response_format=StructuredOutput
        )
        self.assertIsInstance(response, StructuredOutput)
        self.assertIsInstance(response.metadata, CompletionMetadata)

    async def test_batch_processing(self):
        """Test basic batch processing"""
        prompts = [self.test_prompts["simple"]] * 2
        result = await self.handler.process(
            prompts,
            model=self.models["openai"],
            mode="batch"
        )
        self.assertIn('total_prompts', result.metadata)
        self.assertEqual(result.metadata['total_prompts'], len(prompts))
        self.assertEqual(len(result.results), len(prompts))

    async def test_metrics_tracking(self):
        """Test metrics tracking"""
        await self.handler.reset_metrics()
        await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"]
        )
        metrics = await self.handler.get_metrics()
        self.assertGreater(metrics['total_requests'], 0)
        self.assertGreater(metrics['total_tokens'], 0)

    def test_token_encoder(self):
        """Test token encoding"""
        text = "Test message"
        for model in self.models.values():
            count = TokenEncoder.estimate_tokens(text, model)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

    async def test_error_handling(self):
        """Test error handling with mocks"""
        # Define a test error response
        test_error = Exception("Test API Error")
        
        # Reset metrics before test
        await self.handler.reset_metrics()
        
        # Using patch.object is more precise and reliable for this case
        with patch.object(
            self.handler.async_openai_client.chat.completions,
            'create',
            new_callable=AsyncMock,
            side_effect=test_error
        ) as mock_create:
            # Verify that the appropriate exception is raised
            with self.assertRaises(Exception) as context:
                await self.handler.process(
                    "Test prompt",
                    model=self.models["openai"]
                )
                
            # Verify the mock was called the expected number of times (MAX_RETRIES)
            self.assertEqual(mock_create.call_count, RetrySettings.MAX_RETRIES)
            
            # Verify all calls were made with the same parameters
            expected_call = {
                'model': self.models["openai"],
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'Test prompt'}
                ],
                'temperature': 0.7
            }
            
            for call in mock_create.call_args_list:
                self.assertEqual(call.kwargs, expected_call)
            
            # Verify we got the expected error
            self.assertEqual(str(context.exception), "Test API Error")

    async def test_system_message(self):
        """Test system message handling"""
        system_msg = "You are a test assistant"
        response = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"],
            system_message=system_msg
        )
        self.assertIsInstance(response, str)

    async def test_temperature_control(self):
        """Test temperature parameter effect"""
        response = await self.handler.process(
            self.test_prompts["simple"],
            model=self.models["openai"],
            temperature=0
        )
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main()