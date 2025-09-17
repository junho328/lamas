#!/bin/bash

# Qwen/Qwen2.5-Coder-1.5B-Instruct QLoRA 훈련 예제 스크립트
# 간단한 설정으로 빠른 테스트 가능

echo "Starting Qwen QLoRA training with example settings..."

python scripts/train_coding.py \
    --algorithm_name APPO \
    --experiment_name qwen_test \
    --seed 42 \
    --cuda \
    --n_rollout_threads 4 \
    --num_env_steps 100000 \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --n_agents 2 \
    --profile_path scripts/profiles/qwen_coding_duo.json \
    --train_dataset_path envs/coding/train.json \
    --test_dataset_path envs/coding/train.json \
    --lr 5e-7 \
    --critic_lr 5e-4 \
    --ppo_epoch 3 \
    --num_mini_batch 2 \
    --save_interval 25 \
    --log_interval 1 \
    --reward_type binary

echo "Training completed!"
