import openai
from typing import Union
import src.utils.config_parser as config_parser

class GPT():
    def __init__(self, api_key: str, model):
        
        # Check if API key is loaded
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in the environment variables.")

        # Initialize the OpenAI API client
        openai.api_key = api_key
        self.model = model
        print(f"Initalized default OpenAI GPT model: {model}")

    # Define the GPT function
    def get_response(self, prompt: Union[str, list[dict[str, str]]], model: str = None, max_output_tokens: int = 512, temperature: float = 0):
        if not model or model == self.model:
            model = self.model
        else:
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
    message = [{'role': 'user', 'content': "What is the capital of the United States?"}]
    
    args = config_parser.get_combined_config()
    api_key = args["env"]["openai_api_key"]
    gpt_model = GPT(api_key, args["gpt_model"])
    
    response = gpt_model.get_response(message, args["gpt_model"], args["gpt_max_output_tokens"], args["gpt_temperature"])
        
    if response:
        print(f"Response: {response}")
    else:
        raise ValueError("[ERROR]: Failed to get a response from the GPT model.")
