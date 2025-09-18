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
        # Load v0.1.4 split
        print("Loading BigCodeBench v0.1.4 split...")
        dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Show sample structure
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample keys: {list(sample.keys())}")
            print(f"✓ Task ID: {sample.get('task_id', 'N/A')}")
            print(f"✓ Has complete_prompt: {'complete_prompt' in sample}")
            print(f"✓ Has instruct_prompt: {'instruct_prompt' in sample}")
            print(f"✓ Has test: {'test' in sample}")
            print(f"✓ Has entry_point: {'entry_point' in sample}")
            print(f"✓ Has canonical_solution: {'canonical_solution' in sample}")
            print(f"✓ Has code_prompt: {'code_prompt' in sample}")
            print(f"✓ Has libs: {'libs' in sample}")
        
        # Split into train/test manually (80/20 split) with fixed seed
        import random
        import numpy as np
        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        
        total_samples = len(dataset)
        train_size = int(total_samples * 0.8)
        
        # Create reproducible indices
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_dataset = dataset.select(train_indices)
        test_dataset = dataset.select(test_indices)
        print(f"✓ Train split (seed={seed}): {len(train_dataset)} samples")
        print(f"✓ Test split (seed={seed}): {len(test_dataset)} samples")
        
        print("\n✓ All tests passed! BigCodeBench dataset is ready to use.")
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_bigcodebench_loading()
    sys.exit(0 if success else 1)
