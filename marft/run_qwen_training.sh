#!/bin/bash

# Qwen/Qwen2.5-Coder-1.5B-Instruct QLoRA 훈련 스크립트
# BigCodeBench Instruct 데이터셋 사용

# 가상환경 활성화
source ../lamas/bin/activate

# 데이터셋 변환 (Hugging Face에서 직접 로드)
echo "Converting BigCodeBench dataset from Hugging Face..."
python envs/coding/convert_bigcodebench.py \
    --input bigcode/bigcodebench \
    --output data/bigcodebench_instruct_train_marft.json \
    --max_samples 1000 \
    --dataset_type instruct \
    --split train \
    --seed 42

# 테스트 데이터셋도 변환
python envs/coding/convert_bigcodebench.py \
    --input bigcode/bigcodebench \
    --output data/bigcodebench_instruct_test_marft.json \
    --max_samples 200 \
    --dataset_type instruct \
    --split test \
    --seed 42

# GPU 선택 (기본값: 0, 사용법: ./run_qwen_training.sh 1 또는 ./run_qwen_training.sh 0,1)
GPU_ID=${1:-0}
echo "Using GPU(s): $GPU_ID"

# 훈련 실행
echo "Starting Qwen QLoRA training..."

CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train_coding.py \
    --algorithm_name APPO \
    --experiment_name qwen_coding_duo \
    --seed 42 \
    --cuda \
    --n_training_threads 1 \
    --n_rollout_threads 16 \
    --n_eval_rollout_threads 1 \
    --num_env_steps 1000000 \
    --horizon 1 \
    --env_name CODING \
    --dataset_name bigcodebench \
    --train_dataset_path data/bigcodebench_instruct_train_marft.json \
    --test_dataset_path data/bigcodebench_instruct_test_marft.json \
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
    --ppo_epoch 1 \
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
    --wandb_entity lamas-aipr \
    --wandb_run_name qwen-coder-duo-qwen2.5-1.5b \
    --reward_type binary \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_temperature 0.8 \
    --vllm_top_p 0.95

echo "Training completed!"
