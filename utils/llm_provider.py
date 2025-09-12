from config.parser import load_config
from abc import ABC, abstractmethod
import os
import requests
import json


BASE_URL = "https://api.cxhao.com"  # Updated to OpenAI's base URL for GPT-5 compatibility; replace if using a different provider
API_KEY = "sk-TRRdrKYwo1idvAdARhdQhk6DNY5bo0Agnha7foM8IqmeNMUo"  # Replace with your actual API key (get from OpenAI dashboard)
ENDPOINT = "/v1/chat/completions"
def get_api_key(self, model_name: str) -> str:
    """
    Get API key from config.yaml
    """
    config = load_config()
    api_keys = config.get('api_keys', {})
    return api_keys.get(model_name, None)


class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        pass

class TestLLMProvider(LLMProvider):
    def __init__(self, model_name: str = 'gpt-5'):
        self.model_name = model_name
        self.api_url = f"{BASE_URL}{ENDPOINT}"
        self.model_name = model_name
        api_key=get_api_key("openai"),
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, user_prompt, system_prompt = None, **kwargs):
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that helps people find information."
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 1024,
        }
        payload.update(kwargs)

        response = requests.post(
            self.api_url,
            headers=self.headers,
            data=json.dumps(payload),
            timeout=30
        )

        response.raise_for_status()  # 如果不是200，直接抛出HTTPError
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unexpected response format: {data}")
    
class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = 'gpt-5'):
        import openai
        self.api_url = f"{BASE_URL}{ENDPOINT}"
        self.model_name = model_name
        api_key=get_api_key("openai"),
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt, system_prompt = None, **kwargs):
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that helps people find information."
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 1024,
        }
        payload.update(kwargs)

        response = requests.post(
            self.api_url,
            headers=self.headers,
            data=json.dumps(payload),
            timeout=30
        )

        response.raise_for_status()  # 如果不是200，直接抛出HTTPError
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unexpected response format: {data}")
    
class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = 'gemini-2.5-pro'):
        from google.generativeai import generativeai as genai
        self.model_name = model_name
        genai.configure(api_key=get_api_key("google"))

    def generate_response(self, prompt, system_prompt = None):
        return super().generate_response(prompt, system_prompt) 
    
class AnthropicProvider(LLMProvider):  
    def __init__(self, model_name: str = 'claude-sonnet-4-20250514'):
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=get_api_key("anthropic"))

    def generate_response(self, prompt, system_prompt = None):
        return super().generate_response(prompt, system_prompt) 

class GrokProvider(LLMProvider):
    def __init__(self, model_name: str = 'grok-4'):
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=get_api_key("xai"))

    def generate_response(self, prompt, system_prompt = None):
        return super().generate_response(prompt, system_prompt) 


def get_llm_provider(model_name: str) -> LLMProvider:
    if model_name.startswith('gpt-') or model_name.startswith('text-'):
        return OpenAIProvider(model_name)
    elif model_name.startswith('gemini-'):
        return GeminiProvider(model_name)
    elif model_name.startswith('claude-'):
        return AnthropicProvider(model_name)
    elif model_name.startswith('grok-'):
        return GrokProvider(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return None
