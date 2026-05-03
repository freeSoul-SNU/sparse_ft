#!/bin/bash
# Full pipeline: MT-Bench → MMLU → CSR → results.md
# Run with: nohup bash scripts/rapa/run_all.sh &
set -euo pipefail

RAPA="/home1/irteam/rapa"
LMF="${RAPA}/LMFlow"
RESULTS="${RAPA}/results/results.md"
WRESULTS="/workspace/rapa/results.md"

source "${RAPA}/.rapa/bin/activate"
export HF_HOME="${RAPA}/hf_cache"
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY}"

mkdir -p "${RAPA}/results" /workspace/rapa
cd "${LMF}"

METHODS="sift spiel smt s2ft ltsft"

train_and_eval() {
    local TASK=$1 MODEL=$2 DATA=$3 LR_SCHED=$4 EPOCH=$5 MAXSEQ=$6
    echo "============================================"
    echo "[${TASK}] Starting all methods"
    echo "============================================"

    for METHOD in ${METHODS}; do
        CKPT="${RAPA}/checkpoints/${TASK}_${METHOD}"
        LOG="${CKPT}/logs"
        mkdir -p "${CKPT}" "${LOG}"

        echo "[${TASK}] Training: ${METHOD}"
        PORT=$((29600 + RANDOM % 1000))
        deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${PORT} \
            pipeline/train_method.py \
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
        TRAIN_EC=$?

        if [ ${TRAIN_EC} -ne 0 ]; then
            echo "[${TASK}] ${METHOD} TRAIN FAILED (${TRAIN_EC})"
            continue
        fi
        echo "[${TASK}] ${METHOD} training done"

        # Evaluate
        if [ "${TASK}" = "mtbench" ]; then
            echo "[${TASK}] Evaluating ${METHOD} with MT-Bench (vLLM + GPT-4o-mini)"
            python pipeline/eval_mtbench.py \
                --model_path "${CKPT}" --method "${METHOD}" \
                --results_file "${RESULTS}" --judge "gpt-4o-mini" \
                --openai_api_key "${OPENAI_API_KEY}" --num_gpus 8 \
                2>&1 | tee "${LOG}/eval.log"
        elif [ "${TASK}" = "mmlu" ]; then
            echo "[${TASK}] Evaluating ${METHOD} with MMLU 5-shot (vLLM)"
            python pipeline/eval_lmharness.py \
                --model_path "${CKPT}" --method "${METHOD}" \
                --task mmlu --num_fewshot 5 \
                --results_file "${RESULTS}" --num_gpus 8 \
                2>&1 | tee "${LOG}/eval.log"
        elif [ "${TASK}" = "csr" ]; then
            echo "[${TASK}] Evaluating ${METHOD} with CSR 0-shot (vLLM)"
            python pipeline/eval_lmharness.py \
                --model_path "${CKPT}" --method "${METHOD}" \
                --task csr --num_fewshot 0 \
                --results_file "${RESULTS}" --num_gpus 8 \
                2>&1 | tee "${LOG}/eval.log"
        fi
        echo "[${TASK}] ${METHOD} eval done"

        # Copy results to workspace
        cp "${RESULTS}" "${WRESULTS}" 2>/dev/null || true
    done
    echo "[${TASK}] ALL METHODS DONE"
}

# ======== 1. MT-Bench ========
train_and_eval "mtbench" "mistralai/Mistral-7B-v0.3" "${RAPA}/data/oasst1_lmflow.json" "linear" 1 512

# ======== 2. MMLU ========
train_and_eval "mmlu" "meta-llama/Llama-2-7b-hf" "/home1/irteam/datasets/mmlu/mmlu.json" "cosine" 1 512

# ======== 3. CSR ========
train_and_eval "csr" "meta-llama/Llama-2-7b-hf" "/home1/irteam/datasets/merge/merge.json" "cosine" 1 512

echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Results: ${RESULTS}"
echo "============================================"
