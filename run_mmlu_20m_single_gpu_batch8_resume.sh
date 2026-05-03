#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RAPA_HOME="${RAPA_HOME:-/data/nksol0405/LLM/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DATASET="${DATASET:-${RAPA_HOME}/OwLore_Dataset/mmlu/mmlu.json}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/sparse_ft_20m_mmlu_single_gpu}"
RESULTS_FILE="${RESULTS_FILE:-${RESULT_ROOT}/results.md}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"
METHODS="${METHODS:-smt s2ft ltsft}"
GPU_INDEX="${GPU_INDEX:-0}"
TARGET_PARAMS="${TARGET_PARAMS:-20000000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_EVAL="${RUN_EVAL:-true}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
RESET_RESULTS="${RESET_RESULTS:-false}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"

if [ ! -f "${DATASET}" ]; then
    echo "Dataset not found: ${DATASET}"
    exit 1
fi

if [ -f /home/nksol0405/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/nksol0405/anaconda3/etc/profile.d/conda.sh
else
    export PATH="/home/nksol0405/anaconda3/condabin:${PATH}"
    eval "$(conda shell.bash hook)"
fi
conda activate "${ENV_NAME}"

if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    export CUDA_HOME="/usr/local/cuda-12.3"
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export HF_HOME="${HF_HOME:-${RAPA_HOME}/hf_cache}"
export RAPA_HOME
export SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA_HOME}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"

mkdir -p "${RESULT_ROOT}/checkpoints" "${RESULT_ROOT}/logs"
cp -r "${SPARSE_FT_ROOT}/pipeline/." "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/"
cp -r "${SPARSE_FT_ROOT}/configs/." "${LMFLOW_DIR}/configs/rapa/"

if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RESULTS_FILE}" ]; then
    {
        echo "# MMLU 20M Sparse FT Single-GPU Results"
        echo ""
        echo "- model: ${MODEL}"
        echo "- dataset: ${DATASET}"
        echo "- target_params: ${TARGET_PARAMS}"
        echo "- gpu: ${GPU_INDEX}"
        echo ""
    } > "${RESULTS_FILE}"
else
    {
        echo ""
        echo "## Resume Run"
        echo ""
        echo "- methods: ${METHODS}"
        echo "- batch_size: ${BATCH_SIZE}"
        echo "- gradient_accumulation_steps: ${GRAD_ACCUM}"
        echo "- learning_rate: ${LEARNING_RATE}"
        echo "- started_at: $(date --iso-8601=seconds)"
    } >> "${RESULTS_FILE}"
fi

if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${METRICS_FILE}" ]; then
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

cd "${LMFLOW_DIR}"

for method in ${METHODS}; do
    run_name="mmlu_20m_${method}"
    ckpt="${RESULT_ROOT}/checkpoints/${run_name}"
    log_dir="${RESULT_ROOT}/logs/${run_name}"
    mkdir -p "${ckpt}" "${log_dir}"

    extra_args=()
    if [ -n "${MAX_STEPS}" ]; then
        extra_args+=(--max_steps "${MAX_STEPS}")
    fi

    echo "============================================"
    echo "[mmlu] Training ${method} on GPU ${GPU_INDEX} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
    echo "============================================"
    port=$((29500 + RANDOM % 1000))
    train_start="$(date +%s)"
    train_peak_file="${log_dir}/train_peak_memory_mb.txt"
    train_stop_file="${log_dir}/train_peak_memory.stop"
    start_gpu_monitor "${train_peak_file}" "${train_stop_file}"

    set +e
    deepspeed --include=localhost:0 --master_port="${port}" \
        "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py" \
        --method "${method}" \
        --model_name_or_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --output_dir "${ckpt}" \
        --num_train_epochs "${EPOCHS}" \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate "${LEARNING_RATE}" \
        --lr_scheduler_type cosine \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --target_params "${TARGET_PARAMS}" \
        --bf16 \
        --seed 42 \
        "${extra_args[@]}" \
        2>&1 | tee "${log_dir}/train.log"
    train_ec=${PIPESTATUS[0]}
    set -e

    train_elapsed=$(( $(date +%s) - train_start ))
    train_peak_mb="$(stop_gpu_monitor "${train_peak_file}" "${train_stop_file}")"
    echo "[mmlu] ${method} train elapsed_seconds=${train_elapsed} peak_memory_mb=${train_peak_mb} exit=${train_ec}" | tee -a "${log_dir}/train.log"
    echo -e "${method}\ttrain\t${train_elapsed}\t${train_peak_mb}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${train_ec}" >> "${METRICS_FILE}"

    if [ "${train_ec}" -ne 0 ]; then
        echo "| ${method} | train_failed:${train_ec} | | |" >> "${RESULTS_FILE}"
        echo "[mmlu] ${method} train failed with exit=${train_ec}"
        continue
    fi

    if [ "${RUN_EVAL}" = "true" ]; then
        echo "============================================"
        echo "[mmlu] Evaluating ${method}"
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
        echo -e "${method}\teval\t${eval_elapsed}\t${eval_peak_mb}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${eval_ec}" >> "${METRICS_FILE}"
        echo "[mmlu] ${method} eval exit=${eval_ec}"
    fi
done

echo "Done. Results: ${RESULTS_FILE}"
echo "Metrics: ${METRICS_FILE}"
