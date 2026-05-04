#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RAPA_HOME="${RAPA_HOME:-/data/nksol0405/LLM/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/sparse_ft_20m_mmlu_single_gpu}"
RESULTS_FILE="${RESULTS_FILE:-${RESULT_ROOT}/results.md}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"
METHODS="${METHODS:-smt}"
GPU_INDEX="${GPU_INDEX:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"

if [ -f /home/nksol0405/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/nksol0405/anaconda3/etc/profile.d/conda.sh
else
    export PATH="/home/nksol0405/anaconda3/condabin:${PATH}"
    eval "$(conda shell.bash hook)"
fi
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export RAPA_HOME
export SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA_HOME}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${RAPA_HOME}/hf_cache}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "${RESULT_ROOT}/logs"
cp -r "${SPARSE_FT_ROOT}/pipeline/." "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/"

if [ ! -f "${METRICS_FILE}" ]; then
    echo -e "method\tphase\telapsed_seconds\tpeak_memory_mb\tbatch_size\tgradient_accumulation_steps\texit_code" > "${METRICS_FILE}"
fi

gpu_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null \
        | awk 'NR == 1 { gsub(/ /, ""); print int($1); found=1 } END { if (!found) print 0 }'
}

start_gpu_monitor() {
    local peak_file="$1"
    local stop_file="$2"
    rm -f "${stop_file}"
    echo "0" > "${peak_file}"
    (
        peak=0
        while [ ! -f "${stop_file}" ]; do
            mem="$(gpu_used_mb)"
            if [ "${mem}" -gt "${peak}" ]; then
                peak="${mem}"
                echo "${peak}" > "${peak_file}"
            fi
            sleep "${GPU_MONITOR_INTERVAL}"
        done
        mem="$(gpu_used_mb)"
        if [ "${mem}" -gt "${peak}" ]; then
            peak="${mem}"
        fi
        echo "${peak}" > "${peak_file}"
    ) &
    GPU_MONITOR_PID="$!"
}

stop_gpu_monitor() {
    local peak_file="$1"
    local stop_file="$2"
    touch "${stop_file}"
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    cat "${peak_file}"
}

for method in ${METHODS}; do
    run_name="mmlu_20m_${method}"
    ckpt="${RESULT_ROOT}/checkpoints/${run_name}"
    log_dir="${RESULT_ROOT}/logs/${run_name}"
    mkdir -p "${log_dir}"

    if [ ! -d "${ckpt}" ]; then
        echo "Checkpoint not found: ${ckpt}"
        exit 1
    fi

    echo "============================================"
    echo "[mmlu] Evaluating ${method} only"
    echo "============================================"
    eval_start="$(date +%s)"
    eval_peak_file="${log_dir}/eval_peak_memory_mb.txt"
    eval_stop_file="${log_dir}/eval_peak_memory.stop"
    start_gpu_monitor "${eval_peak_file}" "${eval_stop_file}"

    set +e
    python "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/eval_lmharness.py" \
        --model_path "${ckpt}" \
        --method "${method}" \
        --task mmlu \
        --num_fewshot 5 \
        --results_file "${RESULTS_FILE}" \
        --num_gpus 1 \
        2>&1 | tee "${log_dir}/eval.log"
    eval_ec=${PIPESTATUS[0]}
    set -e

    eval_elapsed=$(( $(date +%s) - eval_start ))
    eval_peak_mb="$(stop_gpu_monitor "${eval_peak_file}" "${eval_stop_file}")"
    echo "[mmlu] ${method} eval elapsed_seconds=${eval_elapsed} peak_memory_mb=${eval_peak_mb} exit=${eval_ec}" | tee -a "${log_dir}/eval.log"
    echo -e "${method}\teval\t${eval_elapsed}\t${eval_peak_mb}\t8\t1\t${eval_ec}" >> "${METRICS_FILE}"
done
