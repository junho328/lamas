#!/bin/bash

# Qwen/Qwen2.5-Coder-1.5B-Instruct QLoRA 훈련 스크립트
# BigCodeBench Instruct 데이터셋 사용

# 데이터셋 변환 (Hugging Face에서 직접 로드)
echo "Converting BigCodeBench dataset from Hugging Face..."
python envs/coding/convert_bigcodebench.py \
    --input bigcode/bigcodebench \
    --output data/bigcodebench_marft.json \
    --max_samples 1000 \
    --dataset_type regular \
    --split train

# 훈련 실행
echo "Starting Qwen QLoRA training..."

python scripts/train_coding.py \
    --algorithm_name APPO \
    --experiment_name qwen_coding_duo \
    --seed 42 \
    --cuda \
    --n_training_threads 1 \
    --n_rollout_threads 8 \
    --n_eval_rollout_threads 1 \
    --num_env_steps 1000000 \
    --horizon 1 \
    --env_name CODING \
    --dataset_name bigcodebench \
    --train_dataset_path data/bigcodebench_marft.json \
    --test_dataset_path data/bigcodebench_marft.json \
    --flag train \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --max_new_tokens 512 \
    --n_agents 2 \
    --profile_path scripts/profiles/qwen_coding_duo.json \
    --context_window 2048 \
    --episode_length 200 \
    --warmup_steps 0 \
    --hidden_size 64 \
    --use_orthogonal \
    --lr 5e-7 \
    --critic_lr 5e-4 \
    --opti_eps 1e-5 \
    --weight_decay 0 \
    --gradient_cp_steps 1 \
    --ppo_epoch 5 \
    --use_clipped_value_loss \
    --clip_param 0.2 \
    --num_mini_batch 4 \
    --entropy_coef 0.01 \
    --value_loss_coef 1 \
    --use_max_grad_norm \
    --max_grad_norm 0.5 \
    --use_gae \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --use_proper_time_limits \
    --use_huber_loss \
    --use_value_active_masks \
    --use_policy_active_masks \
    --huber_delta 10.0 \
    --kl_threshold 1e-3 \
    --use_linear_lr_decay \
    --save_interval 50 \
    --log_interval 1 \
    --use_eval \
    --eval_interval 10 \
    --eval_episodes 10 \
    --use_wandb \
    --wandb_project qwen-coding-marft \
    --wandb_entity your-entity \
    --wandb_run_name qwen-coder-duo-qwen2.5-1.5b \
    --reward_type binary

echo "Training completed!"
