#!/bin/bash
# Full pipeline: MT-Bench -> MMLU -> CSR -> results.md
set -euo pipefail

ENV_NAME="${ENV_NAME:-rapa_h200}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_PATH="${SPARSE_FT_ROOT}/$(basename "${BASH_SOURCE[0]}")"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

NUM_GPUS="${NUM_GPUS:-1}"
SLURM_TIME="${SLURM_TIME:-auto}"
maybe_reexec_with_srun "${SCRIPT_PATH}" "$@"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMF="${LMFLOW_DIR:-${RAPA}/LMFlow}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${RAPA}/conda_envs/${ENV_NAME}}"
RESULTS="${RESULTS:-${RAPA}/results/results.md}"
WRESULTS="${WRESULTS:-${RAPA}/results/results.workspace.md}"
MTBENCH_DATASET="${MTBENCH_DATASET:-${RAPA}/data/oasst1_lmflow.json}"
MMLU_DATASET="${MMLU_DATASET:-${RAPA}/OwLore_Dataset/mmlu/mmlu.json}"
CSR_DATASET="${CSR_DATASET:-${RAPA}/OwLore_Dataset/merge/merge.json}"

if [ "${NUM_GPUS}" = "2" ]; then
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1}"
else
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0}"
fi

activate_sparse_ft_conda
configure_cuda_env
export HF_HOME="${RAPA}/hf_cache"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAPA_HOME="${RAPA}"
export SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA}/peft}"
export PYTHONPATH="${LMF}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export DS_CONFIG="${DS_CONFIG:-${LMF}/configs/rapa/ds_zero1.json}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set."
    exit 1
fi

mkdir -p "${RAPA}/results" "${LMF}/src/lmflow/pipeline/rapa" "${LMF}/configs/rapa"
cp -r "${SPARSE_FT_ROOT}/pipeline/." "${LMF}/src/lmflow/pipeline/rapa/"
cp -r "${SPARSE_FT_ROOT}/configs/." "${LMF}/configs/rapa/"
cd "${LMF}"

METHODS="sift spiel smt s2ft ltsft"

train_and_eval() {
    local TASK=$1 MODEL=$2 DATA=$3 LR_SCHED=$4 EPOCH=$5 MAXSEQ=$6
    echo "============================================"
    echo "[${TASK}] Starting all methods"
    echo "============================================"

    if [ ! -f "${DATA}" ]; then
        echo "[${TASK}] dataset not found, skipping: ${DATA}"
        return 0
    fi

    for METHOD in ${METHODS}; do
        CKPT="${RAPA}/checkpoints/${TASK}_${METHOD}"
        LOG="${CKPT}/logs"
        mkdir -p "${CKPT}" "${LOG}"

        echo "[${TASK}] Training: ${METHOD}"
        PORT=$((29600 + RANDOM % 1000))
        set +e
        deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port=${PORT} \
            "${LMF}/src/lmflow/pipeline/rapa/train_method.py" \
            --method ${METHOD} \
            --model_name_or_path ${MODEL} \
            --dataset_path ${DATA} \
            --output_dir ${CKPT} \
            --num_train_epochs ${EPOCH} \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --learning_rate 5e-5 \
            --lr_scheduler_type ${LR_SCHED} \
            --max_seq_length ${MAXSEQ} \
            --target_params 170000000 \
            --bf16 \
            --hf_token ${HF_TOKEN} \
            --seed 42 \
            2>&1 | tee "${LOG}/train.log"
        TRAIN_EC=${PIPESTATUS[0]}
        set -e

        if [ ${TRAIN_EC} -ne 0 ]; then
            echo "[${TASK}] ${METHOD} TRAIN FAILED (${TRAIN_EC})"
            continue
        fi
        echo "[${TASK}] ${METHOD} training done"

        # Evaluate
        if [ "${TASK}" = "mtbench" ]; then
            echo "[${TASK}] Evaluating ${METHOD} with MT-Bench (vLLM + GPT-4o-mini)"
            python "${LMF}/src/lmflow/pipeline/rapa/eval_mtbench.py" \
                --model_path "${CKPT}" --method "${METHOD}" \
                --results_file "${RESULTS}" --judge "gpt-4o-mini" \
                --openai_api_key "${OPENAI_API_KEY}" --num_gpus "${NUM_GPUS}" \
                2>&1 | tee "${LOG}/eval.log"
        elif [ "${TASK}" = "mmlu" ]; then
            echo "[${TASK}] Evaluating ${METHOD} with MMLU 5-shot (vLLM)"
            python "${LMF}/src/lmflow/pipeline/rapa/eval_lmharness.py" \
                --model_path "${CKPT}" --method "${METHOD}" \
                --task mmlu --num_fewshot 5 \
                --results_file "${RESULTS}" --num_gpus "${NUM_GPUS}" \
                2>&1 | tee "${LOG}/eval.log"
        elif [ "${TASK}" = "csr" ]; then
            echo "[${TASK}] Evaluating ${METHOD} with CSR 0-shot (vLLM)"
            python "${LMF}/src/lmflow/pipeline/rapa/eval_lmharness.py" \
                --model_path "${CKPT}" --method "${METHOD}" \
                --task csr --num_fewshot 0 \
                --results_file "${RESULTS}" --num_gpus "${NUM_GPUS}" \
                2>&1 | tee "${LOG}/eval.log"
        fi
        echo "[${TASK}] ${METHOD} eval done"

        # Copy results to workspace
        cp "${RESULTS}" "${WRESULTS}" 2>/dev/null || true
    done
    echo "[${TASK}] ALL METHODS DONE"
}

# ======== 1. MT-Bench ========
train_and_eval "mtbench" "mistralai/Mistral-7B-v0.3" "${MTBENCH_DATASET}" "linear" 1 512

# ======== 2. MMLU ========
train_and_eval "mmlu" "meta-llama/Llama-2-7b-hf" "${MMLU_DATASET}" "cosine" 1 512

# ======== 3. CSR ========
train_and_eval "csr" "meta-llama/Llama-2-7b-hf" "${CSR_DATASET}" "cosine" 1 512

echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Results: ${RESULTS}"
echo "============================================"
