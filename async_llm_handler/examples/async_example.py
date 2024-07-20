# File: async_llm_handler/examples/async_example.py

import asyncio
from async_llm_handler import LLMHandler

async def main():
    handler = LLMHandler()
    
    prompt = "What is the meaning of life?"

    # Using specific models
    models = ['gemini_flash', 'gpt_4o', 'gpt_4o_mini', 'claude_3_5_sonnet', 'claude_3_haiku']
    tasks = [handler.query(prompt, model=model, sync=False) for model in models]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for model, response in zip(models, responses):
        if isinstance(response, Exception):
            print(f"Error with {model}: {str(response)}\n")
        else:
            print(f"{model.replace('_', ' ').title()} Response: {response}\n")

    # Example with max_input_tokens and max_output_tokens
    limited_prompt = "Summarize the entire history of human civilization in great detail."
    try:
        response = await handler.query(limited_prompt, model='gpt_4o', sync=False, max_input_tokens=1000, max_output_tokens=100)
        print(f"GPT-4o Response (limited input to 1000 tokens, output to 100 tokens): {response}\n")
    except Exception as e:
        print(f"Error with GPT-4o (limited tokens): {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(main())