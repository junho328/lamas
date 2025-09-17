#!/usr/bin/env python3
"""
Test script to verify BigCodeBench dataset loading from Hugging Face.
"""

import sys
import os
sys.path.append(".")

from datasets import load_dataset

def test_bigcodebench_loading():
    """Test loading BigCodeBench dataset from Hugging Face."""
    
    print("Testing BigCodeBench dataset loading from Hugging Face...")
    
    try:
        # Load train split
        print("Loading train split...")
        train_dataset = load_dataset("bigcode/bigcodebench", split="train", version="0.1.4")
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        
        # Show sample structure
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"✓ Sample keys: {list(sample.keys())}")
            print(f"✓ Task ID: {sample.get('task_id', 'N/A')}")
            print(f"✓ Has question: {'question' in sample}")
            print(f"✓ Has test: {'test' in sample}")
            print(f"✓ Has entry_point: {'entry_point' in sample}")
        
        # Load test split
        print("\nLoading test split...")
        test_dataset = load_dataset("bigcode/bigcodebench", split="test", version="0.1.4")
        print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
        
        print("\n✓ All tests passed! BigCodeBench dataset is ready to use.")
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_bigcodebench_loading()
    sys.exit(0 if success else 1)
