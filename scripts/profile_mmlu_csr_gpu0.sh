#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/scripts/$(basename "${BASH_SOURCE[0]}")"
source "${REPO_ROOT}/scripts/env_utils.sh"

NUM_GPUS="${NUM_GPUS:-1}"
if [ "${NUM_GPUS}" != "1" ]; then
    echo "$(basename "${SCRIPT_PATH}") profiles one GPU; use NUM_GPUS=1." >&2
    exit 1
fi
SLURM_TIME="${SLURM_TIME:-auto}"
maybe_reexec_with_srun "${SCRIPT_PATH}" "$@"

ENV_NAME="${ENV_NAME:-sparse-ft}"
LLM_ROOT="${LLM_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profiling}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/logs}"
CSV_PATH="${CSV_PATH:-${RESULT_ROOT}/train_profile_gpu0.csv}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
GPU_INDEX="${GPU_INDEX:-0}"
METHODS="${METHODS:-sift spiel smt s2ft ltsft}"
MAX_STEPS="${MAX_STEPS:-}"
LTSFT_MASK_SEARCH_STEPS="${LTSFT_MASK_SEARCH_STEPS:-100}"
LTSFT_N_FT_ITERATIONS="${LTSFT_N_FT_ITERATIONS:-1}"
MMLU_DATASET="${MMLU_DATASET:-${RAPA_HOME}/OwLore_Dataset/mmlu/mmlu.json}"
CSR_DATASET="${CSR_DATASET:-${RAPA_HOME}/OwLore_Dataset/merge/merge.json}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

activate_sparse_ft_conda
configure_cuda_env

export HF_HOME="${HF_HOME:-${RAPA_HOME}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
export RAPA_HOME
export SPIEL_DELTA_DTYPE="${SPIEL_DELTA_DTYPE:-float32}"
export SPIEL_SELECTION_ALGORITHM="${SPIEL_SELECTION_ALGORITHM:-rigl}"
export SPIEL_RESELECTION_STEPS="${SPIEL_RESELECTION_STEPS:-20}"
export SPIEL_SELECTION_ACCUMULATION_STEPS="${SPIEL_SELECTION_ACCUMULATION_STEPS:-5}"
export SPIEL_RESELECTION_RATE_POLICY="${SPIEL_RESELECTION_RATE_POLICY:-linear}"
export SPIEL_INITIAL_RESELECTION_RATE="${SPIEL_INITIAL_RESELECTION_RATE:-0.2}"
export SPIEL_TARGET_MODULES="${SPIEL_TARGET_MODULES:-q_proj,o_proj,v_proj,k_proj,gate_proj,up_proj,down_proj}"
export SPIEL_STRIP_DS_OPTIMIZER="${SPIEL_STRIP_DS_OPTIMIZER:-true}"
export LTSFT_MASK_SEARCH_STEPS LTSFT_N_FT_ITERATIONS
export SPARSE_FT_ROOT="${REPO_ROOT}"
export PEFT_DIR="${PEFT_DIR:-${RAPA_HOME}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"
NVIDIA_SMI_GPU_ID="${NVIDIA_SMI_GPU_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi command was not found."
    exit 1
fi

mkdir -p "${LMFLOW_DIR}/src/lmflow/pipeline/rapa" "${LMFLOW_DIR}/configs/rapa"
cp -r "${REPO_ROOT}/pipeline/." "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/"
cp -r "${REPO_ROOT}/configs/." "${LMFLOW_DIR}/configs/rapa/"

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
        nvidia-smi --id="${NVIDIA_SMI_GPU_ID}" \
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

    if [ ! -f "${dataset}" ]; then
        echo "[${task}] dataset not found, skipping ${method}: ${dataset}"
        return 0
    fi

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
    deepspeed --include=localhost:"${GPU_INDEX}" --master_port="${master_port}" \
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
        --ltsft_mask_search_steps "${LTSFT_MASK_SEARCH_STEPS}" \
        --ltsft_n_ft_iterations "${LTSFT_N_FT_ITERATIONS}" \
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
        "${MMLU_DATASET}" \
        "cosine" \
        "1" \
        "512"
done

for method in ${METHODS}; do
    run_profile \
        "csr" \
        "${method}" \
        "meta-llama/Llama-2-7b-hf" \
        "${CSR_DATASET}" \
        "cosine" \
        "1" \
        "512"
done

echo "Saved profiling summary to ${CSV_PATH}"
