import backoff
import openai
import requests
import json
import time

class OpenAI:
    def __init__(self, model, api_key, temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        
        # 1. NEW v1.0 SYNTAX: Use the Synchronous Client
        # (Notice we use 'OpenAI', not 'AsyncOpenAI')
        self.client = openai.OpenAI(api_key=self.api_key)

    # 2. ROBUST RETRY LOGIC
    # This replaces the manual 'for loop' retry in your async code
    # and keeps your original script's 'backoff' dependency which is cleaner.
    @backoff.on_exception(backoff.expo, 
                          (openai.RateLimitError, 
                           openai.APIConnectionError, 
                           openai.InternalServerError,
                           openai.Timeout), 
                          max_tries=5, 
                          factor=2)
    def create_chat_completion(self, messages):
        try:
            # 3. NEW v1.0 SYNTAX: client.chat.completions.create
            # This blocks until the result is ready (Synchronous)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
            
        except openai.BadRequestError as e:
            print(f"⚠️ Bad Request (likely context length): {e}")
            return ""
        except Exception as e:
            print(f"⚠️ Unexpected Error: {e}")
            raise e
    
class Claude:
    def __init__(self, model, api_key, temperature=0):

        self.Claude_url = "https://api.anthropic.com/v1"
        self.Claude_api_key = api_key
        self.model = model
        self.temperature = temperature

    @backoff.on_exception(backoff.expo, (requests.exceptions.Timeout,requests.exceptions.ConnectionError,requests.exceptions.RequestException), max_tries=5, factor=2, max_time=60)
    def create_chat_completion(self, messages):
        # convert messages to string
        formatted_string = "\n\n{}: {}\n\nAssistant: ".format("Human" if messages[0]["role"] == "user" else "Assistant", messages[0]["content"])
        url = f"{self.Claude_url}/complete"
        headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01", # use updated version
            "x-api-key": self.Claude_api_key,
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "prompt": formatted_string,
            "max_tokens_to_sample": 256,
            "temperature": self.temperature
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        response_json = response.json()

        return response_json['completion'].strip()