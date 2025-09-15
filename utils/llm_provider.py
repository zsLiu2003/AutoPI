from config.parser import load_config
from abc import ABC, abstractmethod
import os
import requests
import json
import time
from utils.logger import get_logger

logger = get_logger(__name__)

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
    def __init__(self, model_name: str = 'gpt-5', max_retries: int = 3, timeout: int = 60):
        self.model_name = model_name
        self.api_url = f"{BASE_URL}{ENDPOINT}"
        self.max_retries = max_retries
        self.timeout = timeout
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

        # 实现重试机制
        for attempt in range(self.max_retries):
            try:
                logger.info(f"API request attempt {attempt + 1}/{self.max_retries} for model {self.model_name}")

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )

                response.raise_for_status()  # 如果不是200，直接抛出HTTPError
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    result = data["choices"][0]["message"]["content"]
                    if not result or not result.strip():
                        logger.warning(f"API returned empty content on attempt {attempt + 1}")
                        if attempt < self.max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            raise ValueError("API returned empty content after all retries")
                    logger.info(f"API request successful on attempt {attempt + 1}")
                    return result
                else:
                    raise ValueError(f"Unexpected response format: {data}")

            except requests.exceptions.Timeout as e:
                logger.warning(f"API request timeout on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：2, 4, 8秒
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed after {self.max_retries} attempts due to timeout")
                    raise

            except requests.exceptions.RequestException as e:
                logger.warning(f"API request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed after {self.max_retries} attempts due to request error")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise

        # 如果所有重试都失败，抛出最后的异常
        raise Exception(f"Failed to get response after {self.max_retries} attempts")

def get_llm_provider(model_name: str) -> LLMProvider:
    # 根据模型使用情况调整超时和重试参数
    if model_name in ["gpt-4", "gpt-5"]:
        # 对于较大的模型使用更长的超时时间
        return TestLLMProvider(model_name, max_retries=3, timeout=90)
    else:
        return TestLLMProvider(model_name, max_retries=3, timeout=60)
