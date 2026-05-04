#!/bin/bash
# Full MT-Bench training + evaluation pipeline
# Model: Mistral-7B-v0.3, Dataset: oasst1
# Methods: SIFT, SpiEL, SMT, S2FT, LT-SFT
set -euo pipefail

ENV_NAME="${ENV_NAME:-rapa_h200}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_PATH="${SPARSE_FT_ROOT}/$(basename "${BASH_SOURCE[0]}")"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

NUM_GPUS="${NUM_GPUS:-1}"
SLURM_TIME="${SLURM_TIME:-auto}"
maybe_reexec_with_srun "${SCRIPT_PATH}" "$@"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_BASE="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_BASE}/LMFlow}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${RAPA_BASE}/conda_envs/${ENV_NAME}}"
DATA="${DATA:-${RAPA_BASE}/data/oasst1_lmflow.json}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
RESULTS="${RESULTS:-${RAPA_BASE}/results/results.md}"

if [ "${NUM_GPUS}" = "2" ]; then
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1}"
else
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0}"
fi

activate_sparse_ft_conda
configure_cuda_env
export HF_HOME="${RAPA_BASE}/hf_cache"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAPA_HOME="${RAPA_BASE}"
export SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA_BASE}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set."
    exit 1
fi

mkdir -p "${RAPA_BASE}/results" "${LMFLOW_DIR}/src/lmflow/pipeline/rapa" "${LMFLOW_DIR}/configs/rapa"
cp -r "${SPARSE_FT_ROOT}/pipeline/." "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/"
cp -r "${SPARSE_FT_ROOT}/configs/." "${LMFLOW_DIR}/configs/rapa/"

cd "${LMFLOW_DIR}"

if [ ! -f "${DATA}" ]; then
    echo "Dataset not found: ${DATA}"
    exit 1
fi

for METHOD in sift spiel smt s2ft ltsft; do
    CKPT="${RAPA_BASE}/checkpoints/mtbench_${METHOD}"
    LOG="${CKPT}/logs"
    mkdir -p "${CKPT}" "${LOG}"

    echo "========================================"
    echo "[MT-Bench] Training: ${METHOD}"
    echo "========================================"

    PORT=$((29600 + RANDOM % 1000))
    set +e
    deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port=${PORT} \
        "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py" \
        --method ${METHOD} \
        --model_name_or_path ${MODEL} \
        --dataset_path ${DATA} \
        --output_dir ${CKPT} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --lr_scheduler_type linear \
        --max_seq_length 512 \
        --target_params 170000000 \
        --bf16 \
        --hf_token ${HF_TOKEN} \
        --seed 42 \
        2>&1 | tee "${LOG}/train.log"
    TRAIN_EXIT=${PIPESTATUS[0]}
    set -e

    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "[MT-Bench] ${METHOD} training FAILED (exit ${TRAIN_EXIT})"
        continue
    fi
    echo "[MT-Bench] ${METHOD} training DONE"

    # Evaluate with MT-Bench using vLLM
    echo "[MT-Bench] Evaluating: ${METHOD}"
    python "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/eval_mtbench.py" \
        --model_path "${CKPT}" \
        --method "${METHOD}" \
        --results_file "${RESULTS}" \
        --judge "gpt-4o-mini" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --num_gpus "${NUM_GPUS}" \
        2>&1 | tee "${LOG}/eval.log"

    echo "[MT-Bench] ${METHOD} evaluation DONE"
done

echo "========================================"
echo "[MT-Bench] ALL METHODS COMPLETE"
echo "Results: ${RESULTS}"
echo "========================================"
