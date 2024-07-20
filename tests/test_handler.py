# File: async_llm_handler/tests/test_handler.py

import pytest
from async_llm_handler import LLMHandler, Config
from async_llm_handler.exceptions import LLMAPIError

@pytest.fixture
def handler():
    return LLMHandler()

def test_query(handler):
    response = handler.query("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_async_query(handler):
    response = await handler._async_query("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0

def test_invalid_model(handler):
    with pytest.raises(ValueError):
        handler.query("Test prompt", model="invalid_model")

@pytest.mark.asyncio
async def test_all_apis_fail(monkeypatch):
    def mock_api_error(*args, **kwargs):
        raise LLMAPIError("API Error")

    handler = LLMHandler()
    for model in ['gemini', 'cohere', 'llama', 'claude', 'openai']:
        monkeypatch.setattr(handler, f'_query_{model}', mock_api_error)

    with pytest.raises(LLMAPIError, match="All LLM APIs failed to respond"):
        await handler._async_query("Test prompt")