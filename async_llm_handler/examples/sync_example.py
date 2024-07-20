# File: async_llm_handler/examples/sync_example.py

from async_llm_handler import LLMHandler

def main():
    handler = LLMHandler()
    
    prompt = "What is the meaning of life?"
    response = handler.query(prompt)
    print(f"Response: {response}")

    # Using a specific model
    response_openai = handler.query(prompt, model='openai')
    print(f"OpenAI Response: {response_openai}")

if __name__ == "__main__":
    main()