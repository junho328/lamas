# Qwen/Qwen2.5-Coder-1.5B-Instruct QLoRA 훈련 가이드

이 가이드는 BigCodeBench Instruct 데이터셋을 사용하여 Qwen 모델을 QLoRA로 훈련하는 방법을 설명합니다.

## 1. 환경 설정

### 1.1 패키지 설치
```bash
# 기본 요구사항 설치
pip install -r requirements.txt

# Qwen QLoRA 훈련을 위한 추가 패키지 설치
pip install -r requirements_qwen.txt
```

### 1.2 CUDA 설정 확인
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## 2. 데이터셋 준비

### 2.1 BigCodeBench 데이터셋 로드
```bash
# Hugging Face에서 직접 BigCodeBench 데이터셋 로드 (v0.1.4)
# 별도 다운로드 불필요 - 스크립트에서 자동으로 로드됩니다
```

### 2.2 데이터셋 변환
```bash
# Hugging Face에서 BigCodeBench 데이터셋을 MARFT 형식으로 변환
python envs/coding/convert_bigcodebench.py \
    --input bigcode/bigcodebench \
    --output data/bigcodebench_marft.json \
    --max_samples 1000 \
    --dataset_type regular \
    --split train

# 테스트 데이터셋도 변환
python envs/coding/convert_bigcodebench.py \
    --input bigcode/bigcodebench \
    --output data/bigcodebench_test_marft.json \
    --max_samples 100 \
    --dataset_type regular \
    --split test

# 또는 로컬 JSON 파일 사용
python envs/coding/convert_bigcodebench.py \
    --input /path/to/bigcodebench.json \
    --output data/bigcodebench_marft.json \
    --max_samples 1000 \
    --dataset_type regular
```

## 3. 훈련 실행

### 3.1 기본 훈련 실행
```bash
# 스크립트 실행 권한 부여
chmod +x run_qwen_training.sh

# 훈련 실행
./run_qwen_training.sh
```

### 3.2 개별 파라미터로 훈련 실행
```bash
python scripts/train_coding.py \
    --algorithm_name APPO \
    --experiment_name qwen_coding_duo \
    --seed 42 \
    --cuda \
    --n_rollout_threads 8 \
    --num_env_steps 1000000 \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --n_agents 2 \
    --profile_path scripts/profiles/qwen_coding_duo.json \
    --train_dataset_path data/bigcodebench_marft.json \
    --test_dataset_path data/bigcodebench_test_marft.json \
    --lr 5e-7 \
    --critic_lr 5e-4 \
    --ppo_epoch 5 \
    --num_mini_batch 4 \
    --use_eval \
    --eval_interval 10 \
    --save_interval 50 \
    --use_wandb \
    --wandb_project qwen-coding-marft \
    --wandb_entity your-entity \
    --reward_type binary
```

## 4. 주요 설정 파라미터

### 4.1 모델 설정
- `model_name_or_path`: Qwen/Qwen2.5-Coder-1.5B-Instruct
- `n_agents`: 2 (analyzer + coder)
- `context_window`: 2048
- `max_new_tokens`: 512

### 4.2 QLoRA 설정
- `load_in_4bit`: true (QLoRA 활성화)
- `bf16`: true (bfloat16 정밀도 사용)

### 4.3 훈련 설정
- `lr`: 5e-7 (학습률)
- `critic_lr`: 5e-4 (크리틱 학습률)
- `ppo_epoch`: 5 (PPO 에포크 수)
- `num_mini_batch`: 4 (미니배치 수)

### 4.4 환경 설정
- `n_rollout_threads`: 8 (병렬 환경 수)
- `episode_length`: 200 (에피소드 길이)
- `reward_type`: binary (이진 보상)

## 5. 모니터링

### 5.1 Wandb 로깅
```bash
# Wandb 로그인
wandb login

# 프로젝트 설정
export WANDB_PROJECT=qwen-coding-marft
export WANDB_ENTITY=your-entity
```

### 5.2 TensorBoard 로깅
```bash
# TensorBoard 실행
tensorboard --logdir results/qwen_coding_duo/Qwen2.5-Coder-1.5B-Instruct/bigcodebench_instruct/APPO/run_1_agent#2_seed42/logs
```

## 6. 체크포인트 관리

### 6.1 모델 저장
- 체크포인트는 `results/` 디렉토리에 자동 저장됩니다
- `save_interval` 파라미터로 저장 주기 조절 가능

### 6.2 훈련 재개
```bash
# 기존 체크포인트에서 훈련 재개
python scripts/train_coding.py \
    --load_path results/qwen_coding_duo/Qwen2.5-Coder-1.5B-Instruct/bigcodebench_instruct/APPO/run_1_agent#2_seed42/checkpoints/steps_0050 \
    # ... 기타 파라미터들
```

## 7. 에이전트 역할

### 7.1 Analyzer 에이전트
- **역할**: 코드 문제 분석 및 접근 방법 제안
- **출력**: 문제 분석 및 해결 방향
- **with_answer**: false (최종 답변 생성하지 않음)

### 7.2 Coder 에이전트
- **역할**: Analyzer의 분석을 바탕으로 실제 코드 구현
- **출력**: 완성된 코드 솔루션
- **with_answer**: true (최종 답변 생성)

## 8. 성능 최적화 팁

### 8.1 메모리 최적화
- `n_rollout_threads`를 GPU 메모리에 맞게 조절
- `num_mini_batch`를 적절히 설정하여 배치 크기 조절

### 8.2 훈련 속도 최적화
- `gradient_cp_steps`를 사용하여 그래디언트 체크포인팅
- `use_linear_lr_decay`로 학습률 스케줄링

### 8.3 안정성 향상
- `kl_threshold`로 KL 발산 제한
- `use_clipped_value_loss`로 값 손실 클리핑

## 9. 문제 해결

### 9.1 CUDA 메모리 부족
```bash
# 배치 크기 줄이기
--n_rollout_threads 4
--num_mini_batch 2

# 그래디언트 체크포인팅 사용
--gradient_cp_steps 2
```

### 9.2 훈련 불안정
```bash
# 학습률 낮추기
--lr 1e-7
--critic_lr 1e-4

# KL 발산 임계값 조절
--kl_threshold 5e-4
```

### 9.3 성능 저하
```bash
# 보상 타입 변경
--reward_type continuous

# 에피소드 길이 조절
--episode_length 100
```

## 10. 결과 분석

훈련 완료 후 다음 파일들을 확인하세요:
- `results/` 디렉토리의 체크포인트
- `logs/` 디렉토리의 로그 파일
- Wandb 대시보드의 훈련 메트릭
- `summary.json` 파일의 요약 통계
