# File: async_llm_handler/examples/async_example.py

import asyncio
from async_llm_handler import LLMHandler

async def main():
    handler = LLMHandler()
    
    prompt = "What is the meaning of life?"
    response = await handler._async_query(prompt)
    print(f"Response: {response}")

    # Using multiple models concurrently
    tasks = [
        handler._async_query(prompt, model='gemini'),
        handler._async_query(prompt, model='openai'),
        handler._async_query(prompt, model='claude')
    ]
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response}")

if __name__ == "__main__":
    asyncio.run(main())