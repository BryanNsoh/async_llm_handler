# Async LLM Handler

Async LLM Handler is a Python package that provides a unified interface for interacting with multiple Language Model APIs asynchronously. It supports Gemini, Claude, OpenAI, Cohere, and Llama (via Groq) APIs.

## Features

- Asynchronous API calls
- Automatic rate limiting
- Easy switching between different LLM providers
- Fallback mechanism when using multiple APIs
- Token counting and prompt clipping utilities

## Installation

You can install the Async LLM Handler using pip:

```bash
pip install async-llm-handler
```

## Usage

First, set up your environment variables in a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
```

Then, you can use the package as follows:

### Synchronous Usage

```python
from async_llm_handler import LLMHandler

handler = LLMHandler()
response = handler.query("What is the meaning of life?")
print(response)
```

### Asynchronous Usage

```python
import asyncio
from async_llm_handler import LLMHandler

async def main():
    handler = LLMHandler()
    response = await handler._async_query("What is the meaning of life?")
    print(response)

asyncio.run(main())
```

## Advanced Usage

You can specify a particular model to use:

```python
response = handler.query("Tell me a joke", model="openai")
```

Or use multiple models concurrently:

```python
import asyncio
from async_llm_handler import LLMHandler

async def main():
    handler = LLMHandler()
    prompt = "What is the best programming language?"
    
    tasks = [
        handler._async_query(prompt, model='gemini'),
        handler._async_query(prompt, model='openai'),
        handler._async_query(prompt, model='claude')
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Response from model {i+1}: {response}")

asyncio.run(main())
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
