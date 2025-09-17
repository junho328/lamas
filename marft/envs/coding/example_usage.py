#!/usr/bin/env python3
"""
보상 방식 설정 예시 스크립트

이 스크립트는 수정된 CodingEnv 클래스의 다양한 보상 방식을 보여줍니다.
"""

import json
import numpy as np
from coding_env import CodingEnv

def create_sample_profiles():
    """샘플 에이전트 프로필 생성"""
    profiles = [
        {
            "role": "coder",
            "with_answer": True,
            "description": "코드를 생성하는 에이전트"
        },
        {
            "role": "reviewer", 
            "with_answer": False,
            "description": "코드를 검토하는 에이전트"
        },
        {
            "role": "tester",
            "with_answer": True,
            "description": "테스트를 수행하는 에이전트"
        }
    ]
    
    with open("sample_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    
    return "sample_profiles.json"

def create_sample_dataset():
    """샘플 데이터셋 생성"""
    dataset = [
        {
            "prompt": [{"content": "두 수를 더하는 함수를 작성하세요."}],
            "reward_model": {
                "ground_truth": {
                    "inputs": ["2\n3", "5\n7", "10\n20"],
                    "outputs": ["5", "12", "30"]
                }
            }
        }
    ]
    
    with open("sample_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    return "sample_dataset.json"

def test_reward_types():
    """다양한 보상 방식 테스트"""
    
    # 샘플 파일 생성
    profile_path = create_sample_profiles()
    dataset_path = create_sample_dataset()
    
    print("=== 보상 방식 테스트 ===\n")
    
    # 1. 연속 보상 (기본값)
    print("1. 연속 보상 환경:")
    env_continuous = CodingEnv(
        rank=0,
        model_name="test_model",
        num_agents=3,
        profile_path=profile_path,
        dataset_path=dataset_path,
        horizon=5,
        mode="train",
        reward_type="continuous"
    )
    print(f"   - 보상 타입: {env_continuous.reward_type}")
    print()
    
    # 2. 이진 보상
    print("2. 이진 보상 환경:")
    env_binary = CodingEnv(
        rank=0,
        model_name="test_model", 
        num_agents=3,
        profile_path=profile_path,
        dataset_path=dataset_path,
        horizon=5,
        mode="train",
        reward_type="binary"
    )
    print(f"   - 보상 타입: {env_binary.reward_type}")
    print()
    
    # 보상 계산 테스트
    print("=== 보상 계산 테스트 ===")
    
    # 샘플 코드 (정답)
    correct_code = """
def add_numbers(a, b):
    return a + b

# 테스트
a = int(input())
b = int(input())
print(add_numbers(a, b))
"""
    
    # 샘플 코드 (부분 정답)
    partial_code = """
def add_numbers(a, b):
    return a + b + 1  # 잘못된 구현

# 테스트  
a = int(input())
b = int(input())
print(add_numbers(a, b))
"""
    
    # 테스트 케이스
    test_cases = {
        "inputs": ["2\n3", "5\n7", "10\n20"],
        "outputs": ["5", "12", "30"]
    }
    
    # 연속 보상 테스트
    print("\n연속 보상 테스트:")
    reward_correct = env_continuous.compute_reward(correct_code, test_cases)
    reward_partial = env_continuous.compute_reward(partial_code, test_cases)
    print(f"   - 정답 코드 보상: {reward_correct}")
    print(f"   - 부분 정답 코드 보상: {reward_partial}")
    
    # 이진 보상 테스트
    print("\n이진 보상 테스트:")
    reward_correct_binary = env_binary.compute_reward(correct_code, test_cases)
    reward_partial_binary = env_binary.compute_reward(partial_code, test_cases)
    print(f"   - 정답 코드 보상: {reward_correct_binary}")
    print(f"   - 부분 정답 코드 보상: {reward_partial_binary}")
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_reward_types()
