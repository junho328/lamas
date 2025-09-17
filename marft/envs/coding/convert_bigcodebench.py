#!/usr/bin/env python3
"""
Convert BigCodeBench dataset to MARFT coding environment format.

This script converts BigCodeBench JSON data to the format expected by the MARFT coding environment.
Supports both regular BigCodeBench and BigCodeBench Instruct datasets.
Can load data from Hugging Face datasets or local JSON files.
The output format includes:
- prompt: The problem description
- reward_model: Contains ground_truth (test cases)
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset


def load_bigcodebench_data(input_source: str, split: str = "train", dataset_type: str = "regular"):
    """
    Load BigCodeBench data from Hugging Face or local file.
    
    Args:
        input_source: Hugging Face dataset name or path to local JSON file
        split: Dataset split (train/test)
        dataset_type: Type of dataset ("regular" or "instruct")
    
    Returns:
        List of data items
    """
    if input_source.startswith("bigcode/") or "/" in input_source and not input_source.endswith(".json"):
        # Load from Hugging Face
        print(f"Loading BigCodeBench {dataset_type} data from Hugging Face: {input_source}")
        try:
            dataset = load_dataset(input_source, split=split, version="0.1.4")
            print(f"Found {len(dataset)} samples in BigCodeBench {dataset_type} dataset")
            return list(dataset)
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            print("Falling back to local file...")
            # Fallback to local file
            with open(input_source, 'r') as f:
                return json.load(f)
    else:
        # Load from local file
        print(f"Loading BigCodeBench {dataset_type} data from local file: {input_source}")
        with open(input_source, 'r') as f:
            bigcodebench_data = json.load(f)
        print(f"Found {len(bigcodebench_data)} samples in BigCodeBench {dataset_type} dataset")
        return bigcodebench_data

def convert_bigcodebench_to_marft(input_source: str, output_file: str, max_samples: int = None, dataset_type: str = "regular", split: str = "train"):
    """
    Convert BigCodeBench dataset to MARFT format.
    
    Args:
        input_source: Hugging Face dataset name or path to BigCodeBench JSON file
        output_file: Path to output MARFT format JSON file
        max_samples: Maximum number of samples to convert (None for all)
        dataset_type: Type of dataset ("regular" or "instruct")
        split: Dataset split (train/test)
    """
    
    bigcodebench_data = load_bigcodebench_data(input_source, split, dataset_type)
    
    marft_data = []
    converted_count = 0
    
    # Handle different data structures
    if dataset_type == "instruct":
        # BigCodeBench Instruct format: list of items
        data_items = bigcodebench_data
    else:
        # Regular BigCodeBench format: dict with task_id as key
        if isinstance(bigcodebench_data, list):
            # If it's a list (from Hugging Face), convert to dict format
            data_items = [(item.get('task_id', f'task_{i}'), item) for i, item in enumerate(bigcodebench_data)]
        else:
            data_items = bigcodebench_data.items()
    
    for item in data_items:
        if max_samples and converted_count >= max_samples:
            break
            
        try:
            if dataset_type == "instruct":
                # BigCodeBench Instruct format
                task_data = item
                task_id = task_data.get('task_id', f'task_{converted_count}')
                problem_description = task_data.get('instruct_prompt', '')
            else:
                # Regular BigCodeBench format
                task_id, task_data = item
                problem_description = task_data.get('question', '')
            
            # Extract test cases from the test field
            test_cases = task_data.get('test', '')
            
            # Create MARFT format entry
            marft_entry = {
                "prompt": [
                    {
                        "content": f"Problem: {problem_description}\n\nPlease implement a solution for this coding problem."
                    }
                ],
                "reward_model": {
                    "ground_truth": test_cases
                },
                "task_id": task_id,
                "entry_point": task_data.get('entry_point', 'task_func'),
                "canonical_solution": task_data.get('canonical_solution', ''),
                "instruct_prompt": task_data.get('instruct_prompt', ''),
                "libs": task_data.get('libs', [])
            }
            
            marft_data.append(marft_entry)
            converted_count += 1
            
        except Exception as e:
            print(f"Error converting task {task_id if 'task_id' in locals() else converted_count}: {e}")
            continue
    
    print(f"Converted {converted_count} samples to MARFT format")
    
    # Save converted data
    with open(output_file, 'w') as f:
        json.dump(marft_data, f, indent=2)
    
    print(f"Saved MARFT format data to {output_file}")
    
    # Print sample entry for verification
    if marft_data:
        print("\nSample converted entry:")
        print(json.dumps(marft_data[0], indent=2)[:500] + "...")


def main():
    parser = argparse.ArgumentParser(description="Convert BigCodeBench to MARFT format")
    parser.add_argument("--input", required=True, 
                       help="Input source: Hugging Face dataset name (e.g., 'bigcode/bigcodebench') or path to local JSON file")
    parser.add_argument("--output", required=True, help="Output MARFT format JSON file")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to convert")
    parser.add_argument("--split", choices=["train", "test"], default="train", 
                       help="Dataset split (train/test)")
    parser.add_argument("--dataset_type", choices=["regular", "instruct"], default="regular",
                       help="Type of BigCodeBench dataset (regular or instruct)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_bigcodebench_to_marft(args.input, args.output, args.max_samples, args.dataset_type, args.split)


if __name__ == "__main__":
    main()

