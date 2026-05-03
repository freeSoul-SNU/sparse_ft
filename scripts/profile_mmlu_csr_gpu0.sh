#!/usr/bin/env bash
set -euo pipefail

RAPA_HOME="${RAPA_HOME:-/home1/irteam/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profiling}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/logs}"
CSV_PATH="${CSV_PATH:-${RESULT_ROOT}/train_profile_gpu0.csv}"
GPU_INDEX="${GPU_INDEX:-0}"
METHODS="${METHODS:-sift spiel smt s2ft ltsft}"
MAX_STEPS="${MAX_STEPS:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

export HF_HOME="${HF_HOME:-${RAPA_HOME}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi command was not found."
    exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Training entrypoint not found: ${TRAIN_SCRIPT}"
    echo "Run scripts/setup_conda_env.sh first, or set TRAIN_SCRIPT explicitly."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set."
    exit 1
fi

if [ ! -f "${CSV_PATH}" ]; then
    echo "timestamp,task,method,model,dataset,elapsed_sec,peak_mem_mib,exit_code,log_file,output_dir" > "${CSV_PATH}"
fi

monitor_gpu_memory() {
    local out_file=$1
    : > "${out_file}"
    while true; do
        nvidia-smi --id="${GPU_INDEX}" \
            --query-gpu=timestamp,memory.used,utilization.gpu \
            --format=csv,noheader,nounits >> "${out_file}"
        sleep 1
    done
}

run_profile() {
    local task=$1
    local method=$2
    local model=$3
    local dataset=$4
    local lr_sched=$5
    local epoch=$6
    local max_seq=$7

    local run_name="${task}_${method}_${TIMESTAMP}"
    local output_dir="${RESULT_ROOT}/checkpoints/${run_name}"
    local log_dir="${LOG_ROOT}/${run_name}"
    local log_file="${log_dir}/train.log"
    local mem_file="${log_dir}/gpu_mem.csv"
    local master_port
    local start_ts
    local end_ts
    local elapsed
    local peak_mem
    local exit_code
    local extra_args=()

    mkdir -p "${output_dir}" "${log_dir}"
    master_port=$((29500 + RANDOM % 1000))
    if [ -n "${MAX_STEPS}" ]; then
        extra_args+=(--max_steps "${MAX_STEPS}")
    fi

    monitor_gpu_memory "${mem_file}" &
    local monitor_pid=$!
    trap 'kill "${monitor_pid}" >/dev/null 2>&1 || true' EXIT

    start_ts=$(date +%s)

    set +e
    deepspeed --include=localhost:0 --master_port="${master_port}" \
        "${TRAIN_SCRIPT}" \
        --method "${method}" \
        --model_name_or_path "${model}" \
        --dataset_path "${dataset}" \
        --output_dir "${output_dir}" \
        --num_train_epochs "${epoch}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --lr_scheduler_type "${lr_sched}" \
        --max_seq_length "${max_seq}" \
        --target_params 170000000 \
        --bf16 \
        --hf_token "${HF_TOKEN}" \
        --seed 42 \
        "${extra_args[@]}" \
        2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}
    set -e

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" 2>/dev/null || true
    trap - EXIT

    peak_mem=$(awk -F',' 'BEGIN{max=0} {gsub(/ /,"",$2); if ($2+0>max) max=$2+0} END{print max+0}' "${mem_file}")

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${TIMESTAMP}" "${task}" "${method}" "${model}" "${dataset}" \
        "${elapsed}" "${peak_mem}" "${exit_code}" "${log_file}" "${output_dir}" >> "${CSV_PATH}"

    echo "[${task}] ${method} elapsed=${elapsed}s peak_mem=${peak_mem}MiB exit=${exit_code}"
    return 0
}

for method in ${METHODS}; do
    run_profile \
        "mmlu" \
        "${method}" \
        "meta-llama/Llama-2-7b-hf" \
        "/home1/irteam/datasets/mmlu/mmlu.json" \
        "cosine" \
        "1" \
        "512"
done

for method in ${METHODS}; do
    run_profile \
        "csr" \
        "${method}" \
        "meta-llama/Llama-2-7b-hf" \
        "/home1/irteam/datasets/merge/merge.json" \
        "cosine" \
        "1" \
        "512"
done

echo "Saved profiling summary to ${CSV_PATH}"
