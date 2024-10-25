# Combined Handler File: async_llm_handler_combined.py

import os
import json
import time
import logging
import asyncio
import re
from typing import List, Dict, Any, Union, Type, Generic, TypeVar, Optional, Literal
from datetime import datetime
from functools import wraps, lru_cache

from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError
import anthropic
from aiolimiter import AsyncLimiter
import tiktoken

import google.generativeai as genai
import instructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

T = TypeVar('T', bound=BaseModel)


# =======================
# OpenAI and Anthropic Handler
# =======================

class BatchResult(BaseModel, Generic[T]):
    """Structure for batch processing results."""
    metadata: Dict[str, Any] = Field(description="Batch processing metadata")
    results: List[Dict[str, Union[str, T]]] = Field(description="List of processed results")
    error_count: int = Field(default=0, description="Number of failed requests")
    success_count: int = Field(default=0, description="Number of successful requests")


class LLMResponse(BaseModel):
    """Structure for individual LLM responses."""
    content: str
    model: str
    tokens_used: int
    processing_time: float


class RetrySettings:
    """Configuration for retry behavior."""
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 300.0
    JITTER_RANGE = 0.1
    RATE_LIMIT_MULTIPLIER = 1.1


class ModelLimits:
    """Rate limits and configurations for different models."""
    DEFAULT_LIMITS = {
        'gpt-4o': {
            'rpm': 5000,
            'tpm': 800000,
            'max_tokens': 16000,
            'context_window': 128000
        },
        'gpt-4o-mini': {
            'rpm': 5000,
            'tpm': 4000000,
            'max_tokens': 16000,
            'context_window': 128000
        },
        'claude-3-5-sonnet-20241022': {
            'rpm': 1000,
            'tpm': 80000,
            'max_tokens': 8000,
            'context_window': 200000
        }
    }

    def __init__(self, custom_limits: Optional[Dict[str, Dict[str, int]]] = None):
        """Initialize model limits, allowing for custom user-defined limits."""
        if custom_limits:
            self.limits = custom_limits
        else:
            self.limits = self.DEFAULT_LIMITS

    def get_model_limits(self, model: str) -> Dict[str, int]:
        """Retrieve limits for a specific model with validation."""
        if model not in self.limits:
            raise ValueError(f"Unsupported model: {model}")
        return self.limits[model]


class TokenEncoder:
    """Handles token encoding and estimation."""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_encoding(model: str) -> Any:
        """Get cached encoding for a model."""
        encoding_map = {
            'gpt-4o': 'o200k_base',
            'gpt-4o-mini': 'o200k_base',
            'claude-3-5-sonnet-20241022': 'cl100k_base'
        }
        if model not in encoding_map:
            raise ValueError(f"Unsupported model: {model}")
        return tiktoken.get_encoding(encoding_map[model])

    @classmethod
    def estimate_tokens(cls, text: str, model: str) -> int:
        """Estimate token count with cached encoding."""
        try:
            encoding = cls.get_encoding(model)
            return len(encoding.encode(text))
        except ValueError as ve:
            logger.warning(f"Token estimation failed for {model}: {str(ve)}")
            raise
        except Exception as e:
            logger.warning(f"Token estimation failed for {model}: {str(e)}")
            return len(text.split()) * 2  # Fallback estimation


def async_retry(max_retries=RetrySettings.MAX_RETRIES):
    """Enhanced retry decorator with sophisticated error handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            backoff = RetrySettings.INITIAL_BACKOFF
            last_error = None

            for attempt in range(max_retries):
                try:
                    return await func(self, *args, **kwargs)
                except RateLimitError as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        break
                        
                    retry_after = float(e.error.get('retry_after', 
                                      self._get_retry_after_seconds(e)))
                    wait_time = max(backoff, retry_after * RetrySettings.RATE_LIMIT_MULTIPLIER)
                    jitter = wait_time * RetrySettings.JITTER_RANGE * (
                        2 * asyncio.get_running_loop().time() % 1 - 1
                    )
                    wait_time += jitter
                    
                    logger.debug(f"Rate limit reached. Retrying in {wait_time:.2f}s "
                               f"(attempt {attempt + 1}/{max_retries})")
                    
                    await self._handle_rate_limit(wait_time, 
                                                kwargs.get('request', {}).get('model'))
                    backoff = min(backoff * 2, RetrySettings.MAX_BACKOFF)
                    
                except (APIError, anthropic.APIError) as e:
                    last_error = e
                    if attempt == max_retries - 1 or 'invalid_request_error' in str(e):
                        break
                        
                    wait_time = backoff + (backoff * RetrySettings.JITTER_RANGE * (
                        2 * asyncio.get_running_loop().time() % 1 - 1
                    ))
                    logger.debug(f"API error. Retrying in {wait_time:.2f}s "
                               f"(attempt {attempt + 1}/{max_retries})")
                    
                    await asyncio.sleep(wait_time)
                    backoff = min(backoff * 2, RetrySettings.MAX_BACKOFF)
                    
                except Exception as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        break
                        
                    wait_time = backoff + (backoff * RetrySettings.JITTER_RANGE * (
                        2 * asyncio.get_running_loop().time() % 1 - 1
                    ))
                    logger.debug(f"Unexpected error. Retrying in {wait_time:.2f}s "
                               f"(attempt {attempt + 1}/{max_retries})")
                    
                    await asyncio.sleep(wait_time)
                    backoff = min(backoff * 2, RetrySettings.MAX_BACKOFF)

            # Increment failed_requests once after max retries
            async with self.lock:
                self._request_metrics['failed_requests'] += 1

            logger.error(f"Max retries ({max_retries}) exceeded. Last error: {last_error}")
            raise last_error

        return wrapper
    return decorator


class LLMAPIHandler:
    """
    Enhanced handler for OpenAI and Anthropic API interactions with improved error handling,
    rate limiting, and monitoring.
    """

    def __init__(self, request_timeout: float = 30.0, custom_rate_limits: Optional[Dict[str, Dict[str, int]]] = None):
        # Initialize clients with timeout
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("CLAUDE_API_KEY")
        
        if not openai_api_key or not anthropic_api_key:
            raise EnvironmentError("Required API keys (OPENAI_API_KEY and CLAUDE_API_KEY) not found in environment variables")
        
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            timeout=request_timeout
        )
        self.anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_key
        )
        self.async_openai_client = AsyncOpenAI(
            api_key=openai_api_key,
            timeout=request_timeout
        )
        self.async_anthropic_client = anthropic.AsyncAnthropic(
            api_key=anthropic_api_key
        )

        # Initialize model limits
        self.model_limits = ModelLimits(custom_limits=custom_rate_limits)

        # Initialize rate limiters
        self.request_limiters = {
            model: AsyncLimiter(limits['rpm'], 60)
            for model, limits in self.model_limits.limits.items()
        }

        self.token_limiters = {
            model: AsyncLimiter(limits['tpm'], 60)
            for model, limits in self.model_limits.limits.items()
        }

        # Rate limit state management
        self._rate_limit_states = {
            model: asyncio.Event() for model in self.model_limits.limits
        }
        for event in self._rate_limit_states.values():
            event.set()

        self.lock = asyncio.Lock()

        # Performance monitoring
        self._request_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_processing_time': 0.0
        }

    async def _handle_rate_limit(self, wait_time: float, model: Optional[str] = None):
        """Coordinated handling of rate limit pauses with cleanup."""
        if model and model in self._rate_limit_states:
            event = self._rate_limit_states[model]
            was_set = event.is_set()
            if was_set:
                event.clear()
            try:
                await asyncio.sleep(wait_time)
            finally:
                if was_set:
                    event.set()
        else:
            await asyncio.sleep(wait_time)

    def _get_retry_after_seconds(self, exception: Union[RateLimitError, APIError]) -> float:
        """Extract retry delay from various types of API errors."""
        try:
            if isinstance(exception, RateLimitError):
                message = str(exception)
                match = re.search(r'Please try again in (\d+(?:\.\d+)?)s', message)
                return float(match.group(1)) if match else RetrySettings.INITIAL_BACKOFF

            if isinstance(exception, APIError):
                retry_after = exception.headers.get('retry-after')
                if retry_after:
                    return float(retry_after)

            return RetrySettings.INITIAL_BACKOFF
        except:
            return RetrySettings.INITIAL_BACKOFF

    def _get_system_message(self, user_system_message: Optional[str], 
                          response_format: Optional[Type[T]]) -> str:
        """Construct appropriate system message based on requirements."""
        if response_format:
            schema = response_format.model_json_schema()
            base_message = f"Answer exclusively in this JSON format: {schema}"
            return f"{base_message}\n{user_system_message}" if user_system_message else base_message
        return user_system_message or "You are a helpful assistant."

    @async_retry()
    async def _async_process_regular(self, 
                                   request: Dict[str, Any], 
                                   response_format: Optional[Type[T]] = None
                                   ) -> Union[str, T, Exception]:
        """Process a single request with enhanced monitoring and error handling."""
        start_time = time.perf_counter()
        model = request['model']

        # Validate model first before any processing
        if model not in self.model_limits.limits:
            raise ValueError(f"Unsupported model: {model}")

        temperature = request.get('temperature', 0.7)
        prompt = request['prompt']
        system_message = self._get_system_message(
            request.get('system_message'), 
            response_format
        )

        # Validate prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string in 'regular' mode.")

        try:
            # Token estimation after model validation
            total_tokens = sum([
                TokenEncoder.estimate_tokens(text, model)
                for text in [prompt, system_message or "", ""]
            ])

            model_limits = self.model_limits.get_model_limits(model)
            if total_tokens > model_limits['context_window']:
                raise ValueError(
                    f"Input exceeds model's context window of {model_limits['context_window']} tokens"
                )

            # Wait for rate limit clearance
            if model in self._rate_limit_states:
                await self._rate_limit_states[model].wait()

            async with self.request_limiters[model]:
                await self.token_limiters[model].acquire(total_tokens)

                logger.debug(f"Processing {model} request (est. tokens: {total_tokens})")

                if model.startswith('gpt-'):
                    response = await self._process_openai_request(
                        model, prompt, system_message, temperature, response_format
                    )
                elif model.startswith('claude-'):
                    response = await self._process_anthropic_request(
                        model, prompt, system_message, temperature, response_format
                    )
                else:
                    raise ValueError(f"Unsupported model: {model}")

                # Update metrics
                async with self.lock:
                    self._request_metrics['total_requests'] += 1
                    self._request_metrics['successful_requests'] += 1
                    self._request_metrics['total_tokens'] += total_tokens
                    self._request_metrics['total_processing_time'] += time.perf_counter() - start_time

                return response

        except Exception as e:
            # Update failure metrics is now handled in the decorator
            raise

    async def _process_openai_request(self,
                                    model: str,
                                    prompt: str,
                                    system_message: Optional[str],
                                    temperature: float,
                                    response_format: Optional[Type[T]]
                                    ) -> Union[str, T]:
        """Handle OpenAI-specific request processing."""
        messages = [{"role": "user", "content": prompt}]
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        if response_format:
            result = await self.async_openai_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format
            )
            return result.choices[0].message.parsed
        else:
            result = await self.async_openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return result.choices[0].message.content

    async def _process_anthropic_request(self,
                                       model: str,
                                       prompt: str,
                                       system_message: Optional[str],
                                       temperature: float,
                                       response_format: Optional[Type[T]]
                                       ) -> Union[str, T]:
        """Handle Anthropic-specific request processing."""
        formatted_prompt = prompt
        if system_message:
            formatted_prompt = f"{system_message}\n\n{prompt}"

        message = await self.async_anthropic_client.messages.create(
            model=model,
            max_tokens=self.model_limits.get_model_limits(model)['max_tokens'],
            temperature=temperature,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        content = message.content[0].text

        if response_format:
            return self._parse_json_response(content, response_format)
        return content

    def _parse_json_response(self, content: str, response_format: Type[T]) -> T:
        """Parse JSON response with enhanced error handling."""
        try:
            # First attempt: direct JSON parsing
            return response_format(**json.loads(content))
        except json.JSONDecodeError:
            # Second attempt: extract JSON from text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                try:
                    json_str = content[json_start:json_end]
                    return response_format(**json.loads(json_str))
                except:
                    pass

            # Third attempt: try to fix common JSON issues
            try:
                fixed_content = self._fix_json_content(content)
                return response_format(**json.loads(fixed_content))
            except:
                raise ValueError(
                    "Failed to extract valid JSON from response after multiple attempts"
                )

    def _fix_json_content(self, content: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Remove any markdown code block markers
        content = re.sub(r'```json\s*|\s*```', '', content)
        # Fix single quotes to double quotes
        content = re.sub(r"'([^']*)':", r'"\1":', content)
        # Remove any leading/trailing whitespace
        content = content.strip()
        return content

    async def process(self,
                     prompts: Union[str, List[str]],
                     model: str = "gpt-4o-mini",
                     system_message: Optional[str] = None,
                     temperature: float = 0.7,
                     mode: str = "regular",
                     response_format: Optional[Type[T]] = None,
                     output_dir: Optional[str] = None,
                     update_interval: int = 60,
                     deduplicate_prompts: bool = False) -> Union[Any, T, BatchResult[T]]:
        """
        Main processing interface with enhanced batching and monitoring.

        Args:
            prompts: Single prompt string or list of prompts
            model: Model identifier to use
            system_message: Optional system message for context
            temperature: Sampling temperature
            mode: "regular", "async_batch", or "openai_batch" processing mode
            response_format: Optional Pydantic model for response structure
            output_dir: Directory for batch processing outputs
            update_interval: Status update interval for batch processing
            deduplicate_prompts: Whether to remove duplicate prompts

        Returns:
            Single response or BatchResult containing all responses
        """
        start_time = time.perf_counter()

        try:
            # Validate model
            if model not in self.model_limits.limits:
                async with self.lock:
                    self._request_metrics['failed_requests'] += 1
                raise ValueError(f"Unsupported model: {model}")

            if mode == "openai_batch":
                # Official OpenAI Batch API processing
                if not isinstance(prompts, list):
                    raise ValueError("Prompts should be a list in 'openai_batch' mode.")

                if deduplicate_prompts:
                    prompts = list(dict.fromkeys(prompts))

                if not prompts:
                    async with self.lock:
                        self._request_metrics['failed_requests'] += 1
                    raise ValueError("Prompt list is empty.")

                batch_requests = self._construct_batch_requests(
                    prompts, model, temperature, system_message, response_format
                )

                return await self._process_openai_batch(
                    batch_requests, response_format, output_dir, update_interval, prompts
                )

            elif mode == "async_batch":
                # Our own asynchronous batch processing
                if not isinstance(prompts, list):
                    raise ValueError("Prompts should be a list in 'async_batch' mode.")

                if deduplicate_prompts:
                    prompts = list(dict.fromkeys(prompts))

                if not prompts:
                    async with self.lock:
                        self._request_metrics['failed_requests'] += 1
                    raise ValueError("Prompt list is empty.")

                # Validate each prompt
                for prompt in prompts:
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError("Each prompt must be a non-empty string in 'async_batch' mode.")

                # Process asynchronously
                batch_requests = [
                    {
                        "model": model,
                        "prompt": prompt,
                        "system_message": system_message,
                        "temperature": temperature
                    }
                    for prompt in prompts
                ]

                batch_results = await asyncio.gather(
                    *[self._async_process_regular(req, response_format) for req in batch_requests],
                    return_exceptions=True
                )

                results = []
                errors = []
                for prompt, result in zip(prompts, batch_results):
                    if isinstance(result, Exception):
                        errors.append({"prompt": prompt, "error": str(result)})
                    else:
                        results.append({"prompt": prompt, "response": result})

                metadata = {
                    "model": model,
                    "total_prompts": len(prompts),
                    "successful_prompts": len(results),
                    "failed_prompts": len(errors),
                    "processing_time": time.perf_counter() - start_time,
                    "errors": errors
                }

                return BatchResult(
                    metadata=metadata,
                    results=results,
                    error_count=len(errors),
                    success_count=len(results)
                )

            elif mode == "regular":
                if not isinstance(prompts, str):
                    raise ValueError("In 'regular' mode, 'prompts' should be a single string.")

                if not prompts.strip():
                    raise ValueError("Prompt must be a non-empty string in 'regular' mode.")

                # Single prompt processing
                request = {
                    "model": model,
                    "prompt": prompts,
                    "system_message": system_message,
                    "temperature": temperature
                }
                return await self._async_process_regular(request, response_format)

            else:
                raise ValueError(f"Invalid mode: {mode}")

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    def _construct_batch_requests(self, prompts: List[str], model: str, temperature: float,
                                  system_message: Optional[str], response_format: Optional[Type[T]]) -> List[Dict[str, Any]]:
        """Construct a list of batch requests for OpenAI Batch API."""
        batch_requests = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Each prompt must be a non-empty string in 'openai_batch' mode.")

            messages = [{"role": "user", "content": prompt}]
            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})

            if response_format:
                schema = response_format
                messages.insert(0, {"role": "system", "content": f"Answer exclusively in this JSON format: {schema}"})

            batch_requests.append({
                "custom_id": f"request_{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
            })
        return batch_requests

    async def _process_openai_batch(self, requests: List[Dict[str, Any]], response_format: Optional[Type[T]],
                                    output_dir: Optional[str], update_interval: int, original_prompts: List[str]) -> BatchResult[T]:
        """Process a batch of requests using the OpenAI Batch API."""
        if not output_dir:
            output_dir = "batch_output"
        os.makedirs(output_dir, exist_ok=True)

        batch_file_path = os.path.join(output_dir, 'batch_input.jsonl')
        with open(batch_file_path, 'w') as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')

        with open(batch_file_path, 'rb') as f:
            batch_file = self.openai_client.files.create(file=f, purpose="batch")

        batch = self.openai_client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        job_metadata = {
            "batch_id": batch.id,
            "input_file_id": batch.input_file_id,
            "status": batch.status,
            "created_at": batch.created_at,
            "last_updated": datetime.now().isoformat(),
            "num_requests": len(requests)
        }

        metadata_file_path = os.path.join(output_dir, f"batch_{batch.id}_metadata.json")
        with open(metadata_file_path, 'w') as f:
            json.dump(job_metadata, f, indent=2)

        start_time = time.time()
        while True:
            current_time = time.time()
            if current_time - start_time >= update_interval:
                batch = self.openai_client.batches.retrieve(batch.id)
                job_metadata.update({
                    "status": batch.status,
                    "last_updated": datetime.now().isoformat()
                })
                with open(metadata_file_path, 'w') as f:
                    json.dump(job_metadata, f, indent=2)
                logger.info(f"Batch status: {batch.status}")
                start_time = current_time

            if batch.status == "completed":
                logger.info("Batch processing completed!")
                break
            elif batch.status in ["failed", "canceled"]:
                logger.error(f"Batch processing {batch.status}.")
                job_metadata["error"] = f"Batch processing {batch.status}"
                with open(metadata_file_path, 'w') as f:
                    json.dump(job_metadata, f, indent=2)
                return BatchResult(metadata=job_metadata, results=[])

            await asyncio.sleep(10)

        output_file_path = os.path.join(output_dir, f"batch_{batch.id}_output.jsonl")
        file_response = self.openai_client.files.content(batch.output_file_id)
        with open(output_file_path, "w") as output_file:
            output_file.write(file_response.text)

        job_metadata.update({
            "status": "completed",
            "last_updated": datetime.now().isoformat(),
            "output_file_path": output_file_path
        })
        with open(metadata_file_path, 'w') as f:
            json.dump(job_metadata, f, indent=2)

        results = []
        with open(output_file_path, 'r') as f:
            for line, original_prompt in zip(f, original_prompts):
                response = json.loads(line)
                body = response['response']['body']
                choices = body.get('choices', [])

                if len(choices) > 0:
                    content = choices[0]['message']['content']

                    if response_format:
                        try:
                            result = response_format(**json.loads(content))
                            results.append({
                                "prompt": original_prompt,
                                "response": result
                            })
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON response for prompt: {original_prompt}")
                    else:
                        results.append({
                            "prompt": original_prompt,
                            "response": content
                        })
                else:
                    logger.error(f"Unexpected response format: {response}")

        return BatchResult(metadata=job_metadata, results=results)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        async with self.lock:
            metrics = self._request_metrics.copy()
            if metrics['successful_requests'] > 0:
                metrics['average_processing_time'] = (
                    metrics['total_processing_time'] / 
                    metrics['successful_requests']
                )
            else:
                metrics['average_processing_time'] = 0.0

            metrics['success_rate'] = (
                metrics['successful_requests'] / 
                metrics['total_requests']
                if metrics['total_requests'] > 0 else 0.0
            )

            return metrics

    async def reset_metrics(self):
        """Reset performance metrics."""
        async with self.lock:
            self._request_metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_tokens': 0,
                'total_processing_time': 0.0
            }


# =======================
# Gemini Handler
# =======================

VALID_MODELS = Literal[
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-8b"
]

class GeminiHandler:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        logger.info("GeminiHandler initialized successfully")

    def _validate_model(self, model: str) -> None:
        valid_models = ["gemini-1.5-pro-002", "gemini-1.5-pro-002", "gemini-1.5-flash-8b"]
        if model not in valid_models:
            raise ValueError(f"Invalid model. Must be one of: {', '.join(valid_models)}")

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: VALID_MODELS = "gemini-1.5-flash-8b",
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        response_format: Optional[Type[T]] = None,
        max_retries: int = 3,
        output_dir: Optional[str] = None,
    ) -> Union[Any, T, List[T]]:
        """
        Process prompts using Gemini models with Instructor for structured output.
        
        Args:
            prompts: Single prompt or list of prompts
            model: Gemini model version to use
            system_message: Optional system message for context
            temperature: Generation temperature (0.0 to 1.0)
            response_format: Pydantic model for structured output (required)
            max_retries: Maximum number of retries for validation failures
            output_dir: Optional directory for saving results
            
        Returns:
            Processed results in specified format
        """
        self._validate_model(model)
        logger.info(f"Processing with model: {model}")
        
        # Convert single prompt to list for unified processing
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        
        try:
            # Initialize Gemini model
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config={"temperature": temperature}
            )
            
            # Patch with Instructor
            client = instructor.from_gemini(
                client=gemini_model,
                mode=instructor.Mode.GEMINI_JSON,
            )
            
            results = []
            total_prompts = len(prompt_list)
            
            for idx, prompt in enumerate(prompt_list, 1):
                try:
                    messages = []
                    if system_message:
                        messages.append({"role": "system", "content": system_message})
                    messages.append({"role": "user", "content": prompt})
                    
                    logger.info(f"Processing prompt {idx}/{total_prompts}")
                    
                    response = client.create(
                        messages=messages,
                        response_model=response_format,
                        max_retries=max_retries
                    )
                    
                    results.append(response)
                    
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"result_{timestamp}_{idx}.json"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'w') as f:
                            f.write(response.model_dump_json(indent=2))
                    
                except Exception as e:
                    logger.error(f"Error processing prompt {idx}: {str(e)}")
                    results.append(None)
            
            if isinstance(prompts, str):
                return results[0] if results else None
                
            return results
                
        except Exception as e:
            logger.error(f"Critical error in process: {str(e)}")
            raise


# =======================
# Test Models
# =======================

class TestResponse(BaseModel):
    """Test response format"""
    summary: str
    keywords: List[str]
    sentiment: str


class UserInfo(BaseModel):
    """Extract user information with validation."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 0 or v > 120:
            raise ValueError("Age must be between 0 and 120")
        return v


class MovieReview(BaseModel):
    """Structured movie review with rating validation."""
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="Brief review summary")
    genres: List[str] = Field(description="List of movie genres")
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v: float) -> float:
        if v < 0 or v > 10:
            raise ValueError("Rating must be between 0 and 10")
        return v


# =======================
# Test Suite
# =======================

async def run_llmapihandler_tests():
    """Run all tests for the LLMAPIHandler"""
    logger.info("Starting LLMAPIHandler tests...")
    
    # Initialize handler
    handler = LLMAPIHandler(request_timeout=60.0)
    
    try:
        # Test 1: Single prompt (regular mode)
        logger.info("\n=== Testing regular mode with single prompt ===")
        single_result = await handler.process(
            prompts="What is the capital of France? Answer in one word.",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7
        )
        logger.info(f"Regular mode result: {single_result}")

        # Test 2: Structured output
        logger.info("\n=== Testing structured output ===")
        structured_prompt = "Analyze this text: 'I love sunny days in Paris!' Return a summary, keywords, and sentiment."
        structured_result = await handler.process(
            prompts=structured_prompt,
            model="claude-3-5-sonnet-20241022",
            response_format=TestResponse,
            temperature=0.5
        )
        logger.info(f"Structured output result: {structured_result}")

        # Test 3: Async batch mode
        logger.info("\n=== Testing async batch mode ===")
        batch_prompts = [
            "What is the capital of France?",
            "What is the capital of Italy?",
            "What is the capital of Spain?"
        ]
        batch_result = await handler.process(
            prompts=batch_prompts,
            model="claude-3-5-sonnet-20241022",
            mode="async_batch"
        )
        logger.info(f"Async batch successes: {batch_result.success_count}")
        logger.info(f"Async batch failures: {batch_result.error_count}")
        
        # Test 4: OpenAI batch mode
        logger.info("\n=== Testing OpenAI batch mode ===")
        openai_batch_result = await handler.process(
            prompts=batch_prompts,
            model="gpt-4o-mini",
            mode="openai_batch",
            output_dir="test_outputs"
        )
        logger.info(f"OpenAI batch metadata: {openai_batch_result.metadata}")

        # Test 5: System message
        logger.info("\n=== Testing with system message ===")
        system_result = await handler.process(
            prompts="Tell me about Paris.",
            model="claude-3-5-sonnet-20241022",
            system_message="You are a travel expert. Keep responses under 50 words."
        )
        logger.info(f"System message result: {system_result}")

        # Test 6: Get and reset metrics
        logger.info("\n=== Testing metrics ===")
        metrics = await handler.get_metrics()
        logger.info(f"Current metrics: {metrics}")
        await handler.reset_metrics()
        reset_metrics = await handler.get_metrics()
        logger.info(f"Metrics after reset: {reset_metrics}")

        logger.info("\nLLMAPIHandler tests completed successfully!")

    except Exception as e:
        logger.error(f"LLMAPIHandler tests failed with error: {str(e)}")
        raise


async def run_geminihandler_tests():
    """Run all tests for the GeminiHandler"""
    logger.info("\nStarting GeminiHandler tests...")
    
    handler = GeminiHandler()
    
    # Define test cases
    test_cases = [
        {
            "model_type": UserInfo,
            "prompt": "Extract information about John who is a 30 year old software engineer",
            "description": "User Information Extraction"
        },
        {
            "model_type": MovieReview,
            "prompt": "Review the movie 'Inception': A mind-bending thriller about dreams within dreams. Amazing special effects. 9/10",
            "description": "Movie Review Processing"
        }
    ]
    
    models = [
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-8b"
    ]
    
    for model in models:
        logger.info(f"\n=== Testing Model: {model} ===")
        
        for test_case in test_cases:
            try:
                logger.info(f"\nRunning {test_case['description']}")
                result = await handler.process(
                    prompts=test_case["prompt"],
                    model=model,
                    temperature=0.7,
                    response_format=test_case["model_type"],
                    max_retries=2,
                    output_dir=f"test_outputs_{model.replace('-', '_')}"
                )
                logger.info(f"Result: {result}")
                
            except Exception as e:
                logger.error(f"Error testing {model} with {test_case['description']}: {str(e)}")
                continue
        
        # Add a batch test for each model
        try:
            logger.info(f"\nRunning Batch Test for {model}")
            batch_prompts = [
                "Extract information about Sarah who is a 25 year old teacher",
                "Extract information about Mike who is a 45 year old doctor",
                "Extract information about Emma who is a 28 year old artist"
            ]
            
            batch_results = await handler.process(
                prompts=batch_prompts,
                model=model,
                temperature=0.7,
                response_format=UserInfo,
                max_retries=2,
                output_dir=f"batch_outputs_{model.replace('-', '_')}"
            )
            
            logger.info("\nBatch processing results:")
            for idx, result in enumerate(batch_results, 1):
                logger.info(f"Batch Result {idx}: {result}")
                
        except Exception as e:
            logger.error(f"Error in batch test for {model}: {str(e)}")
    
    logger.info("\nGeminiHandler tests completed successfully!")


async def run_all_tests():
    """Run all tests for both LLMAPIHandler and GeminiHandler"""
    await run_llmapihandler_tests()
    await run_geminihandler_tests()
    logger.info("\nAll tests completed successfully!")


# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    asyncio.run(run_all_tests())
