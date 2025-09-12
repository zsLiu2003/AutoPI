from config.parser import load_config
from abc import ABC, abstractmethod
import os
import requests
import json

BASE_URL = "https://api.cxhao.com"
API_KEY = "sk-TRRdrKYwo1idvAdARhdQhk6DNY5bo0Agnha7foM8IqmeNMUo"
ENDPOINT = "/v1/chat/completions"

def get_api_key(model_name: str) -> str:
    """Get API key from config.yaml"""
    config = load_config()
    api_keys = config.get('api_keys', {})
    return api_keys.get(model_name, API_KEY)  # 使用默认API_KEY作为fallback

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, user_prompt: str, system_prompt: str = None, **kwargs) -> str:
        pass

class TestLLMProvider(LLMProvider):
    def __init__(self, model_name: str = 'gpt-5'):
        self.model_name = model_name
        self.api_url = f"{BASE_URL}{ENDPOINT}"
        api_key = get_api_key("openai")
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, user_prompt: str, system_prompt: str = None, **kwargs):
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

def get_llm_provider(model_name: str) -> LLMProvider:
    """获取LLM provider，目前只支持TestLLMProvider"""
    return TestLLMProvider(model_name)
