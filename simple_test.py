#!/usr/bin/env python3
"""
Simple test script for LMSYS dataset loader
"""

try:
    from datasets import load_dataset
    print("✓ Hugging Face datasets available")
except ImportError:
    print("✗ Need to install: pip install datasets")
    exit(1)

def simple_test():
    print("Loading LMSYS Chat-1M dataset (5 samples)...")
    
    # Load small sample
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    sample = dataset.select(range(5))
    
    print(f"Dataset columns: {sample.column_names}")
    
    # Test extraction
    queries = []
    for i, item in enumerate(sample):
        print(f"\nItem {i+1}:")
        conversation = item["conversation"]
        print(f"  Conversation type: {type(conversation)}")
        print(f"  First message: {conversation[0]}")
        
        # Extract user query
        if conversation[0]["role"] == "user":
            content = conversation[0]["content"]
            queries.append(content)
            print(f"  ✓ Extracted: {content[:50]}...")
    
    print(f"\nExtracted {len(queries)} user queries")
    return len(queries) > 0

if __name__ == "__main__":
    success = simple_test()
    print(f"Test {'PASSED' if success else 'FAILED'}")