#!/usr/bin/env python3
"""
Test LLM Judge score extraction
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pipeline.evaluator import CombinedEvaluator

def test_score_extraction():
    """Test score extraction from various LLM judge responses"""

    evaluator = CombinedEvaluator()

    test_cases = [
        # Case from actual log
        ("> *Thought for a couple of seconds*\n\n0", 0.0),
        ("0", 0.0),
        ("85", 85.0),
        ("Score: 75", 75.0),
        ("85/100", 85.0),
        ("75%", 75.0),
        ("90 points", 90.0),
        ("65 out of 100", 65.0),
        ("The score is 80 based on analysis.", 80.0),
        ("```\n95\n```", 95.0),
        ("*thinking*\n\n77", 77.0),
        ("No clear score", 0.0),  # Should default to 0
    ]

    print("Testing LLM Judge score extraction...")

    all_passed = True

    for response, expected_score in test_cases:
        try:
            extracted_score = evaluator._extract_score_from_response(response)
            if abs(extracted_score - expected_score) < 0.1:  # Allow small floating point differences
                print(f"✅ '{response[:30]}...' => {extracted_score} (expected {expected_score})")
            else:
                print(f"❌ '{response[:30]}...' => {extracted_score} (expected {expected_score})")
                all_passed = False
        except Exception as e:
            print(f"💥 '{response[:30]}...' => ERROR: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    success = test_score_extraction()
    if success:
        print("\n🎉 All LLM Judge score extraction tests passed!")
        print("The fixes should resolve the evaluation issues.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)