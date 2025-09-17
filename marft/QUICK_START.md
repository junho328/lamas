# Qwen QLoRA 훈련 빠른 시작 가이드

## 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt
pip install -r requirements_qwen.txt
```

## 2. 빠른 테스트 (기존 데이터셋 사용)

```bash
# 기존 train.json 데이터셋으로 빠른 테스트
./run_qwen_example.sh
```

## 3. BigCodeBench 데이터셋 사용

### 3.1 Hugging Face BigCodeBench 데이터셋
```bash
# 데이터셋 변환 (Hugging Face에서 직접 로드)
python envs/coding/convert_bigcodebench.py \
    --input bigcode/bigcodebench \
    --output data/bigcodebench_marft.json \
    --dataset_type regular \
    --split train \
    --max_samples 1000

# 훈련 실행
python scripts/train_coding.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --n_agents 2 \
    --profile_path scripts/profiles/qwen_coding_duo.json \
    --train_dataset_path data/bigcodebench_marft.json \
    --test_dataset_path data/bigcodebench_marft.json \
    --reward_type binary \
    --n_rollout_threads 4 \
    --num_env_steps 100000
```

### 3.2 로컬 BigCodeBench 데이터셋
```bash
# 데이터셋 변환 (로컬 파일 사용)
python envs/coding/convert_bigcodebench.py \
    --input /path/to/bigcodebench.json \
    --output data/bigcodebench_marft.json \
    --dataset_type regular \
    --max_samples 1000

# 훈련 실행
python scripts/train_coding.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --n_agents 2 \
    --profile_path scripts/profiles/qwen_coding_duo.json \
    --train_dataset_path data/bigcodebench_marft.json \
    --test_dataset_path data/bigcodebench_marft.json \
    --reward_type binary \
    --n_rollout_threads 4 \
    --num_env_steps 100000
```

## 4. 주요 파라미터

- `--model_name_or_path`: Qwen/Qwen2.5-Coder-1.5B-Instruct
- `--n_agents`: 2 (analyzer + coder)
- `--profile_path`: scripts/profiles/qwen_coding_duo.json
- `--reward_type`: binary 또는 continuous
- `--n_rollout_threads`: GPU 메모리에 따라 조절 (4-8 권장)
- `--num_env_steps`: 훈련 스텝 수

## 5. 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir results/

# Wandb 사용 시
wandb login
```

## 6. 체크포인트에서 재개

```bash
python scripts/train_coding.py \
    --load_path results/your_experiment/checkpoints/steps_0050 \
    # ... 기타 파라미터들
```
