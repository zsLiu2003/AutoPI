from config.parser import load_config
from abc import ABC, abstractmethod
import os
import requests
import json
import time
from datetime import datetime
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
    def __init__(self, model_name: str = 'gpt-5', max_retries: int = 3, timeout: int = 60, model_type: str = "unknown"):
        self.model_name = model_name
        self.model_type = model_type  # 模型类型：target, auxiliary, gradient等
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
        
        # 创建日志目录
        self.log_dir = "logs/api_interactions"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 根据模型类型创建不同的日志文件
        self.log_file = os.path.join(self.log_dir, f"{self.model_type}_{self.model_name}.jsonl")

    def _log_interaction(self, request_data: dict, response_data: dict, success: bool, error: str = None):
        """记录API交互到JSONL文件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "request": request_data,
            "response": response_data,
            "success": success,
            "error": error
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")

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
            "max_tokens": 8192,
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
                            # 记录失败的交互
                            self._log_interaction(payload, data, False, "API returned empty content after all retries")
                            raise ValueError("API returned empty content after all retries")
                    logger.info(f"API request successful on attempt {attempt + 1}")
                    # 记录成功的交互
                    self._log_interaction(payload, data, True)
                    return result
                else:
                    # 记录失败的交互
                    self._log_interaction(payload, data, False, f"Unexpected response format: {data}")
                    raise ValueError(f"Unexpected response format: {data}")

            except requests.exceptions.Timeout as e:
                logger.warning(f"API request timeout on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：2, 4, 8秒
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed after {self.max_retries} attempts due to timeout")
                    # 记录超时失败的交互
                    self._log_interaction(payload, {}, False, f"Timeout after {self.max_retries} attempts: {e}")
                    raise e

            except requests.exceptions.RequestException as e:
                logger.warning(f"API request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed after {self.max_retries} attempts due to request error")
                    # 记录请求错误的交互
                    self._log_interaction(payload, {}, False, f"Request error after {self.max_retries} attempts: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    # 记录意外错误的交互
                    self._log_interaction(payload, {}, False, f"Unexpected error after {self.max_retries} attempts: {e}")
                    raise

        # 如果所有重试都失败，抛出最后的异常
        raise Exception(f"All {self.max_retries} attempts failed")

def get_llm_provider(model_name: str, model_type: str = "unknown") -> LLMProvider:
    # 根据模型使用情况调整超时和重试参数
    if model_name in ["gpt-4", "gpt-5"]:
        # 对于较大的模型使用更长的超时时间
        return TestLLMProvider(model_name, max_retries=3, timeout=60, model_type=model_type)
    else:
        return TestLLMProvider(model_name, max_retries=3, timeout=60, model_type=model_type)
