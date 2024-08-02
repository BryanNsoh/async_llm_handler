# PowerShell script to set up the test environment for async_llm_handler and prompt_manager packages

# Create the root directory
$rootDir = "C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\package_tests"
New-Item -ItemType Directory -Force -Path $rootDir | Out-Null

# Change to the root directory
Set-Location $rootDir

# Create necessary files
$files = @{
    "requirements.txt"          = @"
async-llm-handler
llm-prompt-manager
python-dotenv
pytest
pytest-asyncio
"@

    ".env"                      = @"
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
"@

    "test_async_llm_handler.py" = @"
import os
import pytest
from dotenv import load_dotenv
from async_llm_handler import LLMHandler, Config

load_dotenv()

@pytest.fixture
def llm_handler():
    config = Config()
    return LLMHandler(config)

@pytest.mark.asyncio
async def test_query_gemini_flash(llm_handler):
    prompt = "What is the capital of France?"
    response = await llm_handler.query(prompt, model="gemini_flash", sync=False)
    assert isinstance(response, str)
    assert "Paris" in response

@pytest.mark.asyncio
async def test_query_gpt_4o(llm_handler):
    prompt = "Explain the concept of gravity in simple terms."
    response = await llm_handler.query(prompt, model="gpt_4o", sync=False)
    assert isinstance(response, str)
    assert len(response) > 50

@pytest.mark.asyncio
async def test_query_claude_3_5_sonnet(llm_handler):
    prompt = "Write a short poem about artificial intelligence."
    response = await llm_handler.query(prompt, model="claude_3_5_sonnet", sync=False)
    assert isinstance(response, str)
    assert "AI" in response or "artificial intelligence" in response.lower()

def test_sync_query(llm_handler):
    prompt = "What is the largest planet in our solar system?"
    response = llm_handler.query(prompt, model="gemini_flash", sync=True)
    assert isinstance(response, str)
    assert "Jupiter" in response

@pytest.mark.asyncio
async def test_rate_limiting(llm_handler):
    prompt = "Hello, world!"
    responses = []
    for _ in range(10):
        response = await llm_handler.query(prompt, model="gemini_flash", sync=False)
        responses.append(response)
    assert len(responses) == 10
    assert all(isinstance(r, str) for r in responses)

@pytest.mark.asyncio
async def test_max_tokens(llm_handler):
    prompt = "Write a very long essay about the history of computers."
    response = await llm_handler.query(prompt, model="gpt_4o", sync=False, max_output_tokens=100)
    assert isinstance(response, str)
    assert len(response.split()) <= 150  # Approximate check, as tokens != words

if __name__ == "__main__":
    pytest.main([__file__])
"@

    "test_prompt_manager.py"    = @"
import os
import pytest
from prompt_manager import PromptManager

@pytest.fixture
def prompt_manager():
    return PromptManager()

def test_get_prompt_direct_text(prompt_manager):
    prompt = prompt_manager.get_prompt("Hello, {{name}}!", name="Alice")
    assert prompt == "Hello, Alice!"

def test_get_prompt_with_multiple_variables(prompt_manager):
    template = "{{greeting}} {{name}}! You are {{age}} years old."
    prompt = prompt_manager.get_prompt(template, greeting="Hi", name="Bob", age=30)
    assert prompt == "Hi Bob! You are 30 years old."

def test_get_prompt_with_file(prompt_manager, tmp_path):
    template_file = tmp_path / "template.txt"
    template_file.write_text("Welcome to {{place}}!")
    
    prompt = prompt_manager.get_prompt(str(template_file), place="Python")
    assert prompt == "Welcome to Python!"

def test_get_prompt_with_nested_variables(prompt_manager):
    template = "{{user.name}} lives in {{user.city}}."
    user_info = {"name": "Charlie", "city": "New York"}
    prompt = prompt_manager.get_prompt(template, user=user_info)
    assert prompt == "Charlie lives in New York."

def test_get_prompt_with_missing_variable(prompt_manager):
    with pytest.raises(ValueError):
        prompt_manager.get_prompt("Hello, {{name}}!")

def test_load_file(prompt_manager, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file.")
    
    content = prompt_manager.load_file(str(test_file))
    assert content == "This is a test file."

def test_get_prompt_with_file_content(prompt_manager, tmp_path):
    template_file = tmp_path / "template.txt"
    template_file.write_text("Main content: {{sub_content}}")
    
    sub_file = tmp_path / "sub.txt"
    sub_file.write_text("This is sub-content.")
    
    prompt = prompt_manager.get_prompt(str(template_file), sub_content=prompt_manager.load_file(str(sub_file)))
    assert prompt == "Main content: This is sub-content."

if __name__ == "__main__":
    pytest.main([__file__])
"@

    "README.md"                 = @"
# Package Testing

This directory contains tests for the `async_llm_handler` and `llm_prompt_manager` packages.

## Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your `.env` file with the necessary API keys.

## Running Tests

To run all tests:
```
pytest
```

To run tests for a specific package:
```
pytest test_async_llm_handler.py
pytest test_prompt_manager.py
```

## Test Coverage

- `test_async_llm_handler.py`: Tests various functionalities of the async_llm_handler package, including different models, sync/async queries, rate limiting, and token limits.
- `test_prompt_manager.py`: Tests the prompt_manager package's ability to handle direct text prompts, file-based prompts, variable substitution, and error cases.

## Notes

- Ensure you have valid API keys in your `.env` file before running the tests.
- Some tests may take longer to run due to API calls. Be patient!
- If you encounter any issues, check your internet connection and API key validity.
"@
}

foreach ($file in $files.Keys) {
    $files[$file] | Out-File -FilePath (Join-Path $rootDir $file) -Encoding utf8
}

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt

Write-Host "Test environment setup complete. You can now run your tests using 'pytest'."
Write-Host "Remember to update the API keys in the .env file before running the tests."