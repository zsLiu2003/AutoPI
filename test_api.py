#!/usr/bin/env python3
"""
Simple API test to verify LLM provider is working correctly
"""

from utils.llm_provider import get_llm_provider
from utils.logger import get_logger

logger = get_logger(__name__)

def test_api():
    """Test basic API functionality"""
    try:
        provider = get_llm_provider("gpt-5")

        system_prompt = "You are a helpful assistant. Always respond with exactly what the user asks for."
        user_prompt = "Please respond with exactly this text: 'API test successful'"

        print("Testing API connection...")
        response = provider.generate_response(user_prompt, system_prompt)

        print(f"Response: {response}")

        if "API test successful" in response:
            print("✓ API test PASSED")
            return True
        else:
            print("✗ API test FAILED - unexpected response")
            return False

    except Exception as e:
        print(f"✗ API test FAILED - error: {e}")
        return False

if __name__ == "__main__":
    test_api()