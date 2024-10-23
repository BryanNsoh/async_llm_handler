# Async LLM Handler

## Installation

`ash
pip install async_llm_handler
`

## Usage

### Basic Usage

`python
from async_llm_handler import LLMAPIHandler

handler = LLMAPIHandler()
response = await handler.process('Your prompt here', model='gpt-4o-mini')
print(response)
`

### Advanced Usage

Provide advanced usage examples with all possible parameter combinations.

## Configuration Options

- **model**: Specify the LLM model to use.
- **system_message**: Custom system message for context.
- **temperature**: Sampling temperature.

## Troubleshooting

- Ensure your API keys are set in the environment variables or in a .env file.
- Check network connectivity.
- Verify model compatibility.
