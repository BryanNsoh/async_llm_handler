# async_llm_handler/examples/async_example.py

import asyncio
import json
from async_llm_handler import Handler
from async_llm_handler.exceptions import LLMAPIError, RateLimitTimeoutError

async def main():
    handler = Handler(rate_limit_timeout=60.0, retry_attempts=5, retry_delay=2.0)
    
    prompt = "What is the meaning of life? Use any JSON format you see fit."

    # Using specific models with JSON mode
    models = ['gemini_flash', 'gpt_4o', 'gpt_4o_mini', 'claude_3_5_sonnet', 'claude_3_haiku']
    tasks = [handler.query(prompt, model=model, json_mode=True) for model in models]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for model, response in zip(models, responses):
        if isinstance(response, Exception):
            print(f"Error with {model}: {str(response)}\n")
        else:
            print(f"{model.replace('_', ' ').title()} Response (JSON mode):")
            try:
                parsed_response = json.loads(response)
                print(json.dumps(parsed_response, indent=2))
            except json.JSONDecodeError:
                print("Failed to parse response as JSON. Raw response:")
                print(response)
            print()

    # Example with max_input_tokens, max_output_tokens, and JSON mode
    limited_prompt = "Summarize the entire history of human civilization in great detail."
    try:
        response = await handler.query(
            limited_prompt,
            model='gpt_4o',
            max_input_tokens=1000,
            max_output_tokens=100,
            json_mode=True
        )
        print(f"GPT-4o Response (limited tokens, JSON mode):")
        try:
            parsed_response = json.loads(response)
            print(json.dumps(parsed_response, indent=2))
        except json.JSONDecodeError:
            print("Failed to parse response as JSON. Raw response:")
            print(response)
        print()
    except Exception as e:
        print(f"Error with GPT-4o (limited tokens): {str(e)}\n")

    # Test rate limiting and retries
    async def test_rate_limiting():
        for i in range(10):  # Attempt to make 10 rapid requests
            try:
                response = await handler.query(f"Test prompt {i}", model='gpt_4o_mini')
                print(f"Request {i + 1} successful")
            except RateLimitTimeoutError as e:
                print(f"Request {i + 1} failed due to rate limiting: {str(e)}")
            except Exception as e:
                print(f"Request {i + 1} failed: {str(e)}")
            await asyncio.sleep(0.1)  # Small delay between requests

    print("Testing rate limiting:")
    await test_rate_limiting()

if __name__ == "__main__":
    asyncio.run(main())