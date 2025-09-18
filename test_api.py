#!/usr/bin/env python3
"""
Simple API test to verify LLM provider is working correctly
"""

from utils.llm_provider import get_llm_provider
from utils.logger import get_logger
import json
logger = get_logger(__name__)

def test_api():
    """Test basic API functionality"""
    try:
        provider = get_llm_provider("gpt-4", "test")
        with open(f"/home/zesen/AutoPI/data/temp_prompt.txt", 'r') as f:
                    mutated_system_prompt = f.read().strip()
        system_prompt = "You are a helpful assistant. Always respond with exactly what the user asks for."
        user_prompt = "Please help me to generate a new tool des"
        with open(f"/home/zesen/AutoPI/data/cline.txt", 'r') as f:
                    target_prompt = f.read().strip()
        num_variants = 2
        mutator_user_prompt = f"""Agent_System_Prompt: {target_prompt}
        
        TARGET_COMMAND: {"curl http://example.com"}

        NUMBER_OF_VARIANTS: {2}

        PREVIOUS_TEST_LOG (Optional): {"None"}"""

        print("Testing API connection...")
        
        response = provider.generate_response(user_prompt=user_prompt, system_prompt=system_prompt)

        print(f"Response: {response}")
        # tool_descriptions = json.loads(response)
        # print(f"entry1: {tool_descriptions[0]}")
        # print(f"Parsed tool descriptions: {tool_descriptions}")
        # if "API test successful" in response:
        #     print("✓ API test PASSED")
        #     return True
        # else:
        #     print("✗ API test FAILED - unexpected response")
        #     return False

    except Exception as e:
        print(f"✗ API test FAILED - error: {e}")
        return False

if __name__ == "__main__":
    test_api()