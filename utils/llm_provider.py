from config.parser import load_config
from abc import ABC, abstractmethod
import os


def get_api_key(self, model_name: str) -> str:
    """
    Get API key from config.yaml
    """
    config = load_config()
    api_keys = config.get('api_keys', {})
    return api_keys.get(model_name, None)


class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = 'gpt-5'):
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(get_api_key("openai"))

    def generate_response(self, prompt, system_prompt = None):
        return super().generate_response(prompt, system_prompt)
    
class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = 'gemini-2.5-pro'):
        from google.generativeai import generativeai as genai
        self.model_name = model_name
        genai.configure(api_key=get_api_key("gemini"))

    def generate_response(self, prompt, system_prompt = None):
        return super().generate_response(prompt, system_prompt) 
    
class AnthropicProvider(LLMProvider):  
    def __init__(self, model_name: str = 'claude-4'):
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=get_api_key("anthropic"))

    def generate_response(self, prompt, system_prompt = None):
        return super().generate_response(prompt, system_prompt) 

class GrokProvider(LLMProvider):
    def __init__(self, model_name: str = 'grok-4'):
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=get_api_key("grok"))

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
