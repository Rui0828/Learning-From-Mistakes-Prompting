import os
import openai
from dotenv import load_dotenv
from typing import Union

class GPT():
    def __init__(self, api_key: str):
        
        # Check if API key is loaded
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in the environment variables.")

        # Initialize the OpenAI API client
        openai.api_key = api_key

    # Define the GPT function
    @staticmethod
    def get_response(prompt: Union[str, list[dict[str, str]]], model: str = "gpt-4o", max_output_tokens: int = 512, temperature: float = 0):
        print(f"Requesting response from OpenAI GPT model: {model}")
        response = openai.chat.completions.create(
            model = model,
            messages = prompt if type(prompt) == list else [{"role": "user", "content": prompt}],
            max_completion_tokens=max_output_tokens,
            temperature = temperature
        ).choices[0].message.content.replace("Assistant:", "").strip()
        
        return response
        


# Test
if __name__ == "__main__":
    # Test the GPT function
    
    # test_mes = [
    #     "What is the capital of the United States?", 
    #     "You are a Chinese-to-Amis language translator. Could you help me translate the following [zh] sentence: 中興大學是一所位於台灣的優質學校。"
    # ]
    
    # for mes in test_mes:
        
    #     print("-" * 100)
    #     print(f"Input message: {mes}")
        
    #     messages = [
    #         {"role": "user", "content": mes}
    #     ]
        
    #     response = gpt(messages)
        
    #     if response:
    #         print(f"Response: {response}")
    #     else:
    #         print("[ERROR]: Failed to get a response from the GPT model.")
        
    
    message = [{'role': 'user', 'content': "What is the capital of the United States?"}]
    
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    gpt_model = GPT(api_key)
    
    response = gpt_model.get_response(message, "gpt-4o")
        
    if response:
        print(f"Response: {response}")
    else:
        raise ValueError("[ERROR]: Failed to get a response from the GPT model.")
