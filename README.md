# Async LLM Handler

Async LLM Handler is a Python package that provides a unified interface for interacting with multiple Language Model APIs, supporting both synchronous and asynchronous operations. It currently supports Gemini, Claude, and OpenAI APIs.

## Features

- Synchronous and asynchronous API calls
- Support for multiple LLM providers:
  - Gemini (model: gemini_flash)
  - Claude (models: claude_3_5_sonnet, claude_3_haiku)
  - OpenAI (models: gpt_4o, gpt_4o_mini)
- Automatic rate limiting for each API
- Token counting and prompt clipping utilities

## Installation

Install the Async LLM Handler using pip:

```bash
pip install async-llm-handler
```

## Configuration

Before using the package, set up your environment variables in a `.env` file in your project's root directory:

```
GEMINI_API_KEY=your_gemini_api_key
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Basic Usage

#### Synchronous

```python
from async_llm_handler import LLMHandler

handler = LLMHandler()

# Using the default model
response = handler.query("What is the capital of France?", sync=True)
print(response)

# Specifying a model
response = handler.query("Explain quantum computing", model="gpt_4o", sync=True)
print(response)
```

#### Asynchronous

```python
import asyncio
from async_llm_handler import LLMHandler

async def main():
    handler = LLMHandler()

    # Using the default model
    response = await handler.query("What is the capital of France?", sync=False)
    print(response)

    # Specifying a model
    response = await handler.query("Explain quantum computing", model="claude_3_5_sonnet", sync=False)
    print(response)

asyncio.run(main())
```

### Advanced Usage

#### Using Multiple Models Concurrently

```python
import asyncio
from async_llm_handler import LLMHandler

async def main():
    handler = LLMHandler()
    prompt = "Explain the theory of relativity"
    
    tasks = [
        handler.query(prompt, model='gemini_flash', sync=False),
        handler.query(prompt, model='gpt_4o', sync=False),
        handler.query(prompt, model='claude_3_5_sonnet', sync=False)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for model, response in zip(['Gemini Flash', 'GPT-4o', 'Claude 3.5 Sonnet'], responses):
        print(f"Response from {model}:")
        print(response)
        print()

asyncio.run(main())
```

#### Limiting Input and Output Tokens

```python
from async_llm_handler import LLMHandler

handler = LLMHandler()

long_prompt = "Provide a detailed explanation of the entire history of artificial intelligence, including all major milestones and breakthroughs."

response = handler.query(long_prompt, model="gpt_4o", sync=True, max_input_tokens=1000, max_output_tokens=500)
print(response)
```

### Supported Models

The package supports the following models:

1. Gemini:
   - `gemini_flash`

2. Claude:
   - `claude_3_5_sonnet`
   - `claude_3_haiku`

3. OpenAI:
   - `gpt_4o`
   - `gpt_4o_mini`

You can specify these models using the `model` parameter in the `query` method.

### Error Handling

The package uses custom exceptions for error handling. Wrap your API calls in try-except blocks to handle potential errors:

```python
from async_llm_handler import LLMHandler
from async_llm_handler.exceptions import LLMAPIError

handler = LLMHandler()

try:
    response = handler.query("What is the meaning of life?", model="gpt_4o", sync=True)
    print(response)
except LLMAPIError as e:
    print(f"An error occurred: {e}")
```

### Rate Limiting

The package automatically handles rate limiting for each API. The current rate limits are:

- Gemini Flash: 30 requests per minute
- Claude 3.5 Sonnet: 5 requests per minute
- Claude 3 Haiku: 5 requests per minute
- GPT-4o: 5 requests per minute
- GPT-4o mini: 5 requests per minute

If you exceed these limits, the package will automatically wait before making the next request.

## Utility Functions

The package includes utility functions for token counting and prompt clipping:

```python
from async_llm_handler.utils import count_tokens, clip_prompt

text = "This is a sample text for token counting."
token_count = count_tokens(text)
print(f"Token count: {token_count}")

long_text = "This is a very long text that needs to be clipped..." * 100
clipped_text = clip_prompt(long_text, max_tokens=50)
print(f"Clipped text: {clipped_text}")
```

These utilities use the `cl100k_base` encoding by default, which is suitable for most modern language models.

## Logging

The package uses Python's built-in logging module. You can configure logging in your application to see debug information, warnings, and errors from the Async LLM Handler:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

This will display INFO level logs and above from the Async LLM Handler.