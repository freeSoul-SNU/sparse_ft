#!/bin/bash
# Smoke test: verify each training method with a tiny subset (20 steps).
# SIFT → torchrun (backward hooks incompatible with deepspeed)
# Others → deepspeed ZeRO-1
set -euo pipefail

RAPA_BASE="/home/nksol0405/LLM/rapa"
LMFLOW_DIR="${RAPA_BASE}/LMFlow"
DATA_DIR="${RAPA_BASE}/data"
CKPT_DIR="${RAPA_BASE}/checkpoints"

source "${RAPA_BASE}/.rapa/bin/activate"
export HF_HOME="${RAPA_BASE}/hf_cache"
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"

METHOD="${1:-sift}"
MODEL="mistralai/Mistral-7B-v0.3"
DATASET="${DATA_DIR}/oasst1_lmflow.json"
CKPT_OUT="${CKPT_DIR}/smoke_${METHOD}"
mkdir -p "${CKPT_OUT}/logs"

echo "=== SMOKE TEST: ${METHOD} ==="

cd "${LMFLOW_DIR}"
MASTER_PORT=$((RANDOM % 10000 + 20000))

COMMON_ARGS="examples/rapa/train_method.py \
    --method ${METHOD} \
    --model_name_or_path ${MODEL} \
    --dataset_path ${DATASET} \
    --output_dir ${CKPT_OUT} \
    --max_steps 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --max_seq_length 512 \
    --target_params 170000000 \
    --bf16 \
    --hf_token ${HF_TOKEN} \
    --seed 42 \
    --smoke_test"

# All methods: deepspeed ZeRO-1, 8 GPUs
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} ${COMMON_ARGS} \
    2>&1 | tee "${CKPT_OUT}/logs/smoke.log"

echo "[SMOKE TEST ${METHOD}] exit=$?"
