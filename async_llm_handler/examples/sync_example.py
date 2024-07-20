# File: async_llm_handler/examples/sync_example.py

from async_llm_handler import LLMHandler

def main():
    handler = LLMHandler()
    
    prompt = "What is the meaning of life?"

    # Using specific models
    models = ['gemini_flash', 'gpt_4o', 'gpt_4o_mini', 'claude_3_5_sonnet', 'claude_3_haiku']
    for model in models:
        try:
            response = handler.query(prompt, model=model, sync=True)
            print(f"{model.replace('_', ' ').title()} Response: {response}\n")
        except Exception as e:
            print(f"Error with {model}: {str(e)}\n")

    # Example with max_input_tokens and max_output_tokens
    limited_prompt = "Summarize the entire history of human civilization in great detail."
    try:
        response = handler.query(limited_prompt, model='gpt_4o', sync=True, max_input_tokens=1000, max_output_tokens=100)
        print(f"GPT-4o Response (limited input to 1000 tokens, output to 100 tokens): {response}\n")
    except Exception as e:
        print(f"Error with GPT-4o (limited tokens): {str(e)}\n")

if __name__ == "__main__":
    main()