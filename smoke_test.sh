#!/bin/bash
# Smoke test: verify each training method with a tiny subset (20 steps).
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_PATH="${SPARSE_FT_ROOT}/$(basename "${BASH_SOURCE[0]}")"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

NUM_GPUS="${NUM_GPUS:-1}"
SLURM_TIME="${SLURM_TIME:-00:30:00}"
maybe_reexec_with_srun "${SCRIPT_PATH}" "$@"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_BASE="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_BASE}/LMFlow}"
DATA_DIR="${RAPA_BASE}/data"
CKPT_DIR="${RAPA_BASE}/checkpoints"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"

GPU_INDEX="${GPU_INDEX:-0}"
if [ "${NUM_GPUS}" = "2" ]; then
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1}"
else
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:${GPU_INDEX}}"
fi

activate_sparse_ft_conda
configure_cuda_env
export HF_HOME="${RAPA_BASE}/hf_cache"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
export RAPA_HOME="${RAPA_BASE}"
export SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA_BASE}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set."
    exit 1
fi

METHOD="${1:-sift}"
MODEL="mistralai/Mistral-7B-v0.3"
DATASET="${DATA_DIR}/oasst1_lmflow.json"
CKPT_OUT="${CKPT_DIR}/smoke_${METHOD}"
mkdir -p "${CKPT_OUT}/logs" "${LMFLOW_DIR}/src/lmflow/pipeline/rapa" "${LMFLOW_DIR}/configs/rapa"
cp -r "${SPARSE_FT_ROOT}/pipeline/." "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/"
cp -r "${SPARSE_FT_ROOT}/configs/." "${LMFLOW_DIR}/configs/rapa/"

echo "=== SMOKE TEST: ${METHOD} ==="

cd "${LMFLOW_DIR}"
MASTER_PORT=$((RANDOM % 10000 + 20000))

COMMON_ARGS="${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py \
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

# All methods: deepspeed ZeRO-1 on the GPU allocation requested by srun.
deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port=${MASTER_PORT} ${COMMON_ARGS} \
    2>&1 | tee "${CKPT_OUT}/logs/smoke.log"

echo "[SMOKE TEST ${METHOD}] exit=$?"
