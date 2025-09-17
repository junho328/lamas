import numpy as np
import json
import random
import re
from typing import Optional
from . import prime_code

# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_path, mode):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def load_profiles(path):
    with open(path, 'r') as file:
        profiles = json.load(file)
    return profiles

class CodingEnv:

    def __init__(self, rank, model_name, num_agents, profile_path, dataset_path, horizon, mode, seed=None, reward_type="binary"):
        
        self.rank = rank
        self.mode = mode
        if seed is not None:
            random.seed(seed)
        self.model_name = model_name
        self.dataset = load_dataset(dataset_path=dataset_path, mode=mode)
        self.profiles = load_profiles(profile_path)
        self.n_agents = num_agents
        assert self.n_agents == len(self.profiles), "Number of agents must match the number of profiles."
        self.max_steps = horizon
        self.step_count = 0
        
        # 보상 설정
        self.reward_type = reward_type  # "continuous" 또는 "binary"
        
        self.problem = None
        self.label = None
        self.current_state = None
        if rank == 0:
            print(f"The {mode} mode environment has {len(self.dataset)} entries in total.")

    def reset(self):
        # Keep sampling until a valid label is found
        while True:
            problem_answer_pair = random.choice(self.dataset)
            # Try to get the final answer label
            label = problem_answer_pair['reward_model']['ground_truth']
            # If label is still None, skip this sample
            if label is None:
                continue
            # Valid sample found
            self.problem = problem_answer_pair["prompt"][0]['content']
            self.label = label
            break

        self.current_state = '<|im_start|>problem: ' + self.problem + "<|im_end|>\n"
        self.history = []
        obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, actions):
        self.step_count += 1
        actions_to_check = []
        self.state_transition(actions)

        for i in range(self.n_agents):
            if self.profiles[i]["with_answer"]:
                actions_to_check.append(actions[i])

        score = 0.0
        for action in actions_to_check:
            # if self._is_correct(action): 
            #     score += 1.0
            score += self.compute_reward(action, self.label)
        score /= len(actions_to_check) # normalize
        
        if score > 0.0 or self.step_count >= self.max_steps:
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
            
        if score == 0.0:
            self.current_state = self.current_state + "judge: The answer is incorrect.\n"
        else:
            self.current_state = self.current_state + "judge: The answer is correct.\n"

        next_obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        # Team reward: give the same episodic score to all agents (picker included)
        rewards = [score for _ in range(self.n_agents)]
        infos = {"state": self.current_state, "gt": self.label, "episodic_return": score}
        return next_obs, rewards, dones, infos

    def state_transition(self, actions):
        for i, action in enumerate(actions):
            self.current_state = self.current_state + self.profiles[i]["role"] + ": " + action + "\n"

    def _is_correct(self, action):
        """이진 보상을 위한 정확성 검사 - 모든 테스트 케이스가 통과해야 True"""
        res = prime_code.compute_score(action, self.label, continuous=False)
        
        if isinstance(res, tuple):
            success, _ = res
            return success  # True/False
        elif isinstance(res, bool):
            return res
        else:
            return False

    def compute_reward(self, solution_str, test_cases):
        """보상 타입에 따라 다른 보상 계산 방식 사용"""
        if self.reward_type == "binary":
            # 이진 보상: 모든 테스트 통과 시 1.0, 아니면 0.0
            return 1.0 if self._is_correct(solution_str) else 0.0
        else:
            # 연속 보상: 테스트 케이스별 성공률
            res = prime_code.compute_score(solution_str, test_cases, continuous=True)

            if isinstance(res, dict):
                return res
            elif isinstance(res, (int, float, bool)):
                return float(res)
            else:
                return float(res[0])

    def seed(self, seed):
        np.random.seed(seed)

    def get_env_info(self):
        env_info = {"n_agents": self.n_agents}
        return env_info
    
    def close(self):
        pass 