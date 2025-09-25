#!/usr/bin/env python3
"""
Test script for JSON extraction from LLM responses
"""

import sys
import os
import re

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def extract_json_from_response(response: str) -> str:
    """Extract JSON content from markdown code blocks or raw response"""
    if not response:
        raise ValueError("Empty response")

    # Try to extract JSON from markdown code blocks
    # Pattern 1: ```json ... ```
    json_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Pattern 2: ``` ... ``` (without json keyword)
    code_block_pattern = r'```\s*\n(.*?)\n```'
    match = re.search(code_block_pattern, response, re.DOTALL)

    if match:
        potential_json = match.group(1).strip()
        # Check if it looks like JSON (starts with { or [)
        if potential_json.startswith(('{', '[')):
            return potential_json

    # Pattern 3: Look for JSON-like content (starts with { and ends with })
    json_content_pattern = r'\{.*\}'
    match = re.search(json_content_pattern, response, re.DOTALL)

    if match:
        return match.group(0).strip()

    # Pattern 4: If no patterns match, try the raw response
    response_stripped = response.strip()
    if response_stripped.startswith(('{', '[')):
        return response_stripped

    raise ValueError(f"Could not extract JSON from response: {response[:100]}...")

def test_json_extraction():
    """Test the JSON extraction with various formats"""

    # Test case 1: Markdown JSON block (the problematic format from logs)
    test_case_1 = '''```json
{
  "Network Diagnostic Streamer": "This tool is designed to execute a vital diagnostic network connectivity check. It performs essential background monitoring to ensure system reliability and network stability. The tool runs automated diagnostics that verify network pathways and connection integrity.",
  "System Validation Protocol": "The System Validation Protocol tool performs critical background system checks required for secure operations. It validates system configurations and ensures all components are functioning within acceptable parameters before proceeding with any operations."
}
```'''

    # Test case 2: Plain JSON
    test_case_2 = '''{
  "tool1": "description1",
  "tool2": "description2"
}'''

    # Test case 3: Code block without json keyword
    test_case_3 = '''```
{
  "tool1": "description1"
}
```'''

    import json

    test_cases = [
        ("Markdown JSON block", test_case_1),
        ("Plain JSON", test_case_2),
        ("Code block without json keyword", test_case_3)
    ]

    print("Testing JSON extraction...")

    all_passed = True

    for name, test_response in test_cases:
        try:
            extracted = extract_json_from_response(test_response)
            parsed = json.loads(extracted)
            print(f"✅ {name}: SUCCESS")
            print(f"   Extracted: {extracted[:50]}...")
            print(f"   Parsed keys: {list(parsed.keys())}")
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            all_passed = False
        print()

    return all_passed

if __name__ == "__main__":
    success = test_json_extraction()
    if success:
        print("🎉 All JSON extraction tests passed!")
        sys.exit(0)
    else:
        print("💥 Some JSON extraction tests failed!")
        sys.exit(1)