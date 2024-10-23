# /src/utils/llm_api_handler.py

import os
import json
import time
import logging
import asyncio
import re
from typing import List, Dict, Any, Union, Type, Generic, TypeVar, Optional
from datetime import datetime
from functools import wraps, lru_cache
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError
import anthropic
from aiolimiter import AsyncLimiter
import tiktoken

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("CLAUDE_API_KEY")

if not openai_api_key or not anthropic_api_key:
    raise EnvironmentError("Required API keys not found in environment variables")

T = TypeVar('T', bound=BaseModel)

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
    LIMITS = {
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

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """Retrieve limits for a specific model with validation."""
        if model not in cls.LIMITS:
            raise ValueError(f"Unsupported model: {model}")
        return cls.LIMITS[model]

class TokenEncoder:
    """Handles token encoding and estimation."""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_encoding(model: str) -> Any:
        """Get cached encoding for a model."""
        # Validate model first
        if model not in ModelLimits.LIMITS:
            raise ValueError(f"Unsupported model: {model}")
            
        encoding_map = {
            'gpt-4o': 'o200k_base',
            'gpt-4o-mini': 'o200k_base',
            'claude-3-5-sonnet-20241022': 'cl100k_base'
        }
        return tiktoken.get_encoding(encoding_map[model])

    @classmethod
    def estimate_tokens(cls, text: str, model: str) -> int:
        """Estimate token count with cached encoding."""
        try:
            encoding = cls.get_encoding(model)
            return len(encoding.encode(text))
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

            logger.error(f"Max retries ({max_retries}) exceeded. Last error: {last_error}")
            raise last_error

        return wrapper
    return decorator

class LLMAPIHandler:
    """
    Enhanced handler for OpenAI and Anthropic API interactions with improved error handling,
    rate limiting, and monitoring.
    """

    def __init__(self, request_timeout: float = 30.0):
        # Initialize clients with timeout
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
        
        # Initialize rate limiters
        self.request_limiters = {
            model: AsyncLimiter(limits['rpm'], 60)
            for model, limits in ModelLimits.LIMITS.items()
        }
        
        self.token_limiters = {
            model: AsyncLimiter(limits['tpm'], 60)
            for model, limits in ModelLimits.LIMITS.items()
        }
        
        # Rate limit state management
        self._rate_limit_states = {
            model: asyncio.Event() for model in ModelLimits.LIMITS
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
        if model not in ModelLimits.LIMITS:
            raise ValueError(f"Unsupported model: {model}")
            
        temperature = request.get('temperature', 0.7)
        prompt = request['prompt']
        system_message = self._get_system_message(
            request.get('system_message'), 
            response_format
        )

        try:
            # Token estimation after model validation
            total_tokens = sum([
                TokenEncoder.estimate_tokens(text, model)
                for text in [prompt, system_message or "", ""]
            ])
            
            model_limits = ModelLimits.get_model_limits(model)
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
            # Update failure metrics
            async with self.lock:
                self._request_metrics['total_requests'] += 1
                self._request_metrics['failed_requests'] += 1
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
            max_tokens=ModelLimits.get_model_limits(model)['max_tokens'],
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
                     deduplicate_prompts: bool = False,
                     batch_size: int = 50) -> Union[Any, T, BatchResult[T]]:
        """
        Main processing interface with enhanced batching and monitoring.
        
        Args:
            prompts: Single prompt string or list of prompts
            model: Model identifier to use
            system_message: Optional system message for context
            temperature: Sampling temperature
            mode: "regular" or "batch" processing mode
            response_format: Optional Pydantic model for response structure
            output_dir: Directory for batch processing outputs
            update_interval: Status update interval for batch processing
            deduplicate_prompts: Whether to remove duplicate prompts
            batch_size: Size of batches for concurrent processing
            
        Returns:
            Single response or BatchResult containing all responses
        """
        start_time = time.perf_counter()
        
        try:
            if isinstance(prompts, str):
                # Single prompt processing
                request = {
                    "model": model,
                    "prompt": prompts,
                    "system_message": system_message,
                    "temperature": temperature
                }
                return await self._async_process_regular(request, response_format)
            
            elif isinstance(prompts, list) and mode == "batch":
                # Batch processing
                if deduplicate_prompts:
                    prompts = list(dict.fromkeys(prompts))
                
                total_prompts = len(prompts)
                job_id = f"batch_{int(time.time())}_{model}"
                
                logger.info(f"Starting batch job {job_id} with {total_prompts} prompts")
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Process in batches
                results = []
                errors = []
                
                for i in range(0, total_prompts, batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    batch_requests = [
                        {
                            "model": model,
                            "prompt": prompt,
                            "system_message": system_message,
                            "temperature": temperature
                        }
                        for prompt in batch_prompts
                    ]
                    
                    # Process batch concurrently
                    batch_results = await asyncio.gather(
                        *[self._async_process_regular(req, response_format) 
                          for req in batch_requests],
                        return_exceptions=True
                    )
                    
                    # Process results and track errors
                    for prompt, result in zip(batch_prompts, batch_results):
                        if isinstance(result, Exception):
                            errors.append({
                                "prompt": prompt,
                                "error": str(result)
                            })
                        else:
                            results.append({
                                "prompt": prompt,
                                "response": result
                            })
                    
                    # Progress update
                    processed = len(results) + len(errors)
                    logger.info(f"Processed {processed}/{total_prompts} prompts "
                              f"({len(errors)} errors)")
                    
                    if output_dir:
                        self._save_batch_progress(
                            output_dir,
                            job_id,
                            results,
                            errors,
                            processed,
                            total_prompts
                        )
                
                # Final batch result
                metadata = {
                    "job_id": job_id,
                    "model": model,
                    "total_prompts": total_prompts,
                    "successful_prompts": len(results),
                    "failed_prompts": len(errors),
                    "processing_time": time.perf_counter() - start_time,
                    "errors": errors
                }
                
                if output_dir:
                    self._save_batch_results(output_dir, job_id, metadata, results)
                
                return BatchResult(
                    metadata=metadata,
                    results=results,
                    error_count=len(errors),
                    success_count=len(results)
                )
            
            else:
                raise ValueError(
                    "Invalid input: 'prompts' should be a string for regular mode "
                    "or a list for batch mode"
                )
                
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    def _save_batch_progress(self,
                           output_dir: str,
                           job_id: str,
                           results: List[Dict],
                           errors: List[Dict],
                           processed: int,
                           total: int):
        """Save batch processing progress to disk."""
        progress_file = os.path.join(output_dir, f"{job_id}_progress.json")
        progress_data = {
            "processed": processed,
            "total": total,
            "success_count": len(results),
            "error_count": len(errors),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    def _save_batch_results(self,
                          output_dir: str,
                          job_id: str,
                          metadata: Dict,
                          results: List[Dict]):
        """Save final batch results to disk."""
        results_file = os.path.join(output_dir, f"{job_id}_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": metadata,
                "results": results
            }, f, indent=2)

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
            
            
            