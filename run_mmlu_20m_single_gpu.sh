#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_PATH="${SPARSE_FT_ROOT}/$(basename "${BASH_SOURCE[0]}")"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

NUM_GPUS="${NUM_GPUS:-1}"
if [ "${NUM_GPUS}" != "1" ]; then
    echo "$(basename "${SCRIPT_PATH}") is a single-GPU script; use NUM_GPUS=1." >&2
    exit 1
fi
SLURM_TIME="${SLURM_TIME:-auto}"
maybe_reexec_with_srun "${SCRIPT_PATH}" "$@"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${RAPA_HOME}}"
CACHE_ROOT="${CACHE_ROOT:-${RAPA_HOME}/hf_cache}"
TMPDIR="${TMPDIR:-${RAPA_HOME}/tmp}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DATASET="${DATASET:-${RAPA_HOME}/OwLore_Dataset/mmlu/mmlu.json}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/sparse_ft_20m_mmlu_single_gpu}"
RESULTS_FILE="${RESULTS_FILE:-${RESULT_ROOT}/results.md}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"
METHODS="${METHODS:-sift spiel smt s2ft ltsft}"
GPU_INDEX="${GPU_INDEX:-0}"
TARGET_PARAMS="${TARGET_PARAMS:-20000000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
RESET_RESULTS="${RESET_RESULTS:-false}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
GPU_MONITOR_LOG_INTERVAL="${GPU_MONITOR_LOG_INTERVAL:-60}"
SYNC_LMFLOW="${SYNC_LMFLOW:-auto}"
RESUME_TRAINING="${RESUME_TRAINING:-false}"
CPU_OFFLOAD_METHODS="${CPU_OFFLOAD_METHODS:-}"
BASE_DS_CONFIG="${BASE_DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"
OFFLOAD_DS_CONFIG="${OFFLOAD_DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero2_offload.json}"
SIFT_USE_GRADIENT_CALIBRATION="${SIFT_USE_GRADIENT_CALIBRATION:-true}"
SIFT_CALIBRATION_ONLY="${SIFT_CALIBRATION_ONLY:-false}"
SIFT_CALIBRATION_STEPS="${SIFT_CALIBRATION_STEPS:-1}"
SIFT_CALIBRATION_BATCH_SIZE="${SIFT_CALIBRATION_BATCH_SIZE:-1}"
SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-100}"
SMT_CALIBRATION_BATCH_SIZE="${SMT_CALIBRATION_BATCH_SIZE:-1}"
S2FT_CALIBRATION_STEPS="${S2FT_CALIBRATION_STEPS:-100}"
S2FT_CALIBRATION_BATCH_SIZE="${S2FT_CALIBRATION_BATCH_SIZE:-1}"
LTSFT_MASK_SEARCH_STEPS="${LTSFT_MASK_SEARCH_STEPS:-100}"
LTSFT_N_FT_ITERATIONS="${LTSFT_N_FT_ITERATIONS:-1}"
export SIFT_CALIBRATION_STEPS SIFT_CALIBRATION_BATCH_SIZE
export RESUME_TRAINING SIFT_USE_GRADIENT_CALIBRATION SIFT_CALIBRATION_ONLY
export SMT_CALIBRATION_STEPS SMT_CALIBRATION_BATCH_SIZE
export S2FT_CALIBRATION_STEPS S2FT_CALIBRATION_BATCH_SIZE
export LTSFT_MASK_SEARCH_STEPS LTSFT_N_FT_ITERATIONS

if [ ! -f "${DATASET}" ]; then
    echo "Dataset not found: ${DATASET}"
    exit 1
fi

mkdir -p "${SCRATCH_ROOT}" "${CACHE_ROOT}" "${TMPDIR}"
export TMPDIR

activate_sparse_ft_conda

configure_cuda_env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
NVIDIA_SMI_GPU_ID="${NVIDIA_SMI_GPU_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
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

sync_lmflow_sources() {
    local target_dir="${LMFLOW_DIR}/src/lmflow/pipeline/rapa"
    local target_config_dir="${LMFLOW_DIR}/configs/rapa"
    local avail_kb

    if [ "${SYNC_LMFLOW}" = "false" ]; then
        echo "[sync] skipping LMFlow sync (SYNC_LMFLOW=false)"
        return 0
    fi

    avail_kb="$(df -Pk "${LMFLOW_DIR}" | awk 'NR == 2 { print int($4) }')"
    if [ "${SYNC_LMFLOW}" = "auto" ] && [ "${avail_kb}" -lt 65536 ]; then
        echo "[sync] skipping LMFlow sync: only ${avail_kb}KB available on $(df -Pk "${LMFLOW_DIR}" | awk 'NR == 2 { print $6 }')"
        echo "[sync] using existing LMFlow copy. Set SYNC_LMFLOW=true after freeing space to force sync."
        return 0
    fi

    cp -r "${SPARSE_FT_ROOT}/pipeline/." "${target_dir}/"
    cp -r "${SPARSE_FT_ROOT}/configs/." "${target_config_dir}/"
}

sync_lmflow_sources

method_uses_offload() {
    local method="$1"
    local item
    for item in ${CPU_OFFLOAD_METHODS}; do
        if [ "${item}" = "${method}" ]; then
            return 0
        fi
    done
    return 1
}

if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RESULTS_FILE}" ]; then
    {
        echo "# MMLU 20M Sparse FT Single-GPU Results"
        echo ""
        echo "- model: ${MODEL}"
        echo "- dataset: ${DATASET}"
        echo "- target_params: ${TARGET_PARAMS}"
        echo "- gpu: ${GPU_INDEX}"
        echo "- sift_calibration_steps: ${SIFT_CALIBRATION_STEPS}"
        echo "- smt_calibration_steps: ${SMT_CALIBRATION_STEPS}"
        echo "- s2ft_calibration_steps: ${S2FT_CALIBRATION_STEPS}"
        echo "- ltsft_mask_search_steps: ${LTSFT_MASK_SEARCH_STEPS}"
        echo "- ltsft_n_ft_iterations: ${LTSFT_N_FT_ITERATIONS}"
        echo "- smt_calibration_batch_size: ${SMT_CALIBRATION_BATCH_SIZE}"
        echo ""
    } > "${RESULTS_FILE}"
else
    {
        echo ""
        echo "## New Run"
        echo ""
        echo "- methods: ${METHODS}"
        echo "- run_train: ${RUN_TRAIN}"
        echo "- run_eval: ${RUN_EVAL}"
        echo "- batch_size: ${BATCH_SIZE}"
        echo "- gradient_accumulation_steps: ${GRAD_ACCUM}"
        echo "- learning_rate: ${LEARNING_RATE}"
        echo "- resume_training: ${RESUME_TRAINING}"
        echo "- sift_use_gradient_calibration: ${SIFT_USE_GRADIENT_CALIBRATION}"
        echo "- sift_calibration_only: ${SIFT_CALIBRATION_ONLY}"
        echo "- sift_calibration_steps: ${SIFT_CALIBRATION_STEPS}"
        echo "- smt_calibration_steps: ${SMT_CALIBRATION_STEPS}"
        echo "- s2ft_calibration_steps: ${S2FT_CALIBRATION_STEPS}"
        echo "- ltsft_mask_search_steps: ${LTSFT_MASK_SEARCH_STEPS}"
        echo "- ltsft_n_ft_iterations: ${LTSFT_N_FT_ITERATIONS}"
        echo "- smt_calibration_batch_size: ${SMT_CALIBRATION_BATCH_SIZE}"
        echo "- started_at: $(date --iso-8601=seconds)"
    } >> "${RESULTS_FILE}"
fi

if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${METRICS_FILE}" ]; then
    echo -e "method\tphase\telapsed_seconds\tpeak_memory_mb\tpeak_cpu_rss_mb\tbatch_size\tgradient_accumulation_steps\texit_code" > "${METRICS_FILE}"
fi

log_metric_value() {
    local log_file="$1"
    local key="$2"
    grep -E "${key}=" "${log_file}" 2>/dev/null | tail -n 1 | awk -F'=' '{gsub(/[^0-9.]/, "", $2); print $2}'
}

gpu_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${NVIDIA_SMI_GPU_ID}" 2>/dev/null \
        | awk 'NR == 1 { gsub(/ /, ""); print int($1); found=1 } END { if (!found) print 0 }'
}

descendant_pids() {
    local root_pid="$1"
    local pending="${root_pid}"
    local all="${root_pid}"
    local next
    local pid
    local children

    while [ -n "${pending}" ]; do
        next=""
        for pid in ${pending}; do
            children="$(pgrep -P "${pid}" 2>/dev/null || true)"
            if [ -n "${children}" ]; then
                next="${next} ${children}"
                all="${all} ${children}"
            fi
        done
        pending="${next}"
    done

    echo "${all}"
}

job_gpu_used_mb() {
    local root_pid="$1"
    local pids

    pids="$(descendant_pids "${root_pid}")"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v pids="${pids}" '
            BEGIN {
                split(pids, a, " ");
                for (i in a) wanted[a[i]] = 1;
                sum = 0;
            }
            {
                gsub(/ /, "", $1);
                gsub(/ /, "", $2);
                if ($1 in wanted) sum += $2;
            }
            END { print int(sum); }
        '
}

job_cpu_rss_mb() {
    local root_pid="$1"
    local pids csv

    pids="$(descendant_pids "${root_pid}")"
    csv="$(echo "${pids}" | tr ' ' ',')"
    ps -o rss= -p "${csv}" 2>/dev/null | awk 'BEGIN{s=0} {s+=$1} END{print int(s/1024)}'
}

marker_cpu_rss_mb() {
    local marker="$1"
    ps -eo rss=,args= 2>/dev/null \
        | awk -v marker="${marker}" 'index($0, marker) > 0 {sum += $1} END {print int(sum / 1024)}'
}

start_gpu_monitor() {
    local peak_file="$1"
    local stop_file="$2"
    local status_log="${3:-}"
    local phase="${4:-phase}"
    local method="${5:-unknown}"
    local start_ts="${6:-$(date +%s)}"
    local root_pid="${7:-$$}"
    local marker="${8:-}"
    local cpu_peak_file="${peak_file%.txt}_cpu_rss_mb.txt"
    rm -f "${stop_file}"
    echo "0" > "${peak_file}"
    echo "0" > "${cpu_peak_file}"
    (
        peak=0
        cpu_peak=0
        last_log=0
        while [ ! -f "${stop_file}" ]; do
            mem="$(gpu_used_mb)"
            cpu_mem="$(job_cpu_rss_mb "${root_pid}")"
            marker_cpu=0
            if [ -n "${marker}" ]; then
                marker_cpu="$(marker_cpu_rss_mb "${marker}")"
            fi
            if [ "${marker_cpu}" -gt "${cpu_mem}" ]; then
                cpu_mem="${marker_cpu}"
            fi
            if [ "${mem}" -gt "${peak}" ]; then
                peak="${mem}"
                echo "${peak}" > "${peak_file}"
            fi
            if [ "${cpu_mem}" -gt "${cpu_peak}" ]; then
                cpu_peak="${cpu_mem}"
                echo "${cpu_peak}" > "${cpu_peak_file}"
            fi
            now="$(date +%s)"
            if [ -n "${status_log}" ] && [ $((now - last_log)) -ge "${GPU_MONITOR_LOG_INTERVAL}" ]; then
                elapsed=$((now - start_ts))
                job_mem="$(job_gpu_used_mb "${root_pid}")"
                echo "[mmlu] ${method} ${phase} status elapsed_seconds=${elapsed} gpu_memory_mb=${mem} job_gpu_memory_mb=${job_mem} peak_memory_mb=${peak} cpu_rss_mb=${cpu_mem} peak_cpu_rss_mb=${cpu_peak}" | tee -a "${status_log}"
                last_log="${now}"
            fi
            sleep "${GPU_MONITOR_INTERVAL}"
        done
        mem="$(gpu_used_mb)"
        cpu_mem="$(job_cpu_rss_mb "${root_pid}")"
        marker_cpu=0
        if [ -n "${marker}" ]; then
            marker_cpu="$(marker_cpu_rss_mb "${marker}")"
        fi
        if [ "${marker_cpu}" -gt "${cpu_mem}" ]; then
            cpu_mem="${marker_cpu}"
        fi
        if [ "${mem}" -gt "${peak}" ]; then
            peak="${mem}"
        fi
        if [ "${cpu_mem}" -gt "${cpu_peak}" ]; then
            cpu_peak="${cpu_mem}"
        fi
        echo "${peak}" > "${peak_file}"
        echo "${cpu_peak}" > "${cpu_peak_file}"
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

cpu_peak_file_for() {
    local peak_file="$1"
    echo "${peak_file%.txt}_cpu_rss_mb.txt"
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

    train_ec=0
    if [ "${RUN_TRAIN}" = "true" ]; then
        echo "============================================"
        echo "[mmlu] Training ${method} on GPU ${GPU_INDEX} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
        echo "============================================"
        if method_uses_offload "${method}"; then
            train_ds_config="${OFFLOAD_DS_CONFIG}"
            echo "[mmlu] ${method} using CPU optimizer offload with unchanged batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
        else
            train_ds_config="${BASE_DS_CONFIG}"
        fi
        port=$((29500 + RANDOM % 1000))
        train_start="$(date +%s)"
        train_peak_file="${log_dir}/train_peak_memory_mb.txt"
        train_stop_file="${log_dir}/train_peak_memory.stop"
        set +e
        (
            DS_CONFIG="${train_ds_config}" deepspeed --include=localhost:"${GPU_INDEX}" --master_port="${port}" \
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
                --sift_calibration_steps "${SIFT_CALIBRATION_STEPS}" \
                --sift_calibration_batch_size "${SIFT_CALIBRATION_BATCH_SIZE}" \
                --smt_calibration_steps "${SMT_CALIBRATION_STEPS}" \
                --smt_calibration_batch_size "${SMT_CALIBRATION_BATCH_SIZE}" \
                --s2ft_calibration_steps "${S2FT_CALIBRATION_STEPS}" \
                --s2ft_calibration_batch_size "${S2FT_CALIBRATION_BATCH_SIZE}" \
                --ltsft_mask_search_steps "${LTSFT_MASK_SEARCH_STEPS}" \
                --ltsft_n_ft_iterations "${LTSFT_N_FT_ITERATIONS}" \
                --bf16 \
                --seed 42 \
                "${extra_args[@]}"
        ) > >(tee "${log_dir}/train.log") 2>&1 &
        train_pid="$!"
        start_gpu_monitor "${train_peak_file}" "${train_stop_file}" "${log_dir}/train.log" "train" "${method}" "${train_start}" "${train_pid}" "${ckpt}"
        wait "${train_pid}"
        train_ec=$?
        set -e

        train_elapsed=$(( $(date +%s) - train_start ))
        train_peak_mb="$(stop_gpu_monitor "${train_peak_file}" "${train_stop_file}")"
        train_cpu_peak_file="$(cpu_peak_file_for "${train_peak_file}")"
        train_peak_cpu_mb="$(cat "${train_cpu_peak_file}" 2>/dev/null || echo 0)"
        echo "[mmlu] ${method} train elapsed_seconds=${train_elapsed} peak_memory_mb=${train_peak_mb} peak_cpu_rss_mb=${train_peak_cpu_mb} exit=${train_ec}" | tee -a "${log_dir}/train.log"
        echo -e "${method}\ttrain\t${train_elapsed}\t${train_peak_mb}\t${train_peak_cpu_mb}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${train_ec}" >> "${METRICS_FILE}"
        if [ "${method}" = "sift" ]; then
            calibration_elapsed="$(log_metric_value "${log_dir}/train.log" "calibration_seconds" || true)"
            calibration_peak_gpu="$(log_metric_value "${log_dir}/train.log" "calibration_peak_reserved_mb" || true)"
            calibration_peak_cpu="$(log_metric_value "${log_dir}/train.log" "calibration_peak_cpu_rss_mb" || true)"
            if [ -n "${calibration_elapsed}" ]; then
                echo -e "${method}\tcalibration\t${calibration_elapsed}\t${calibration_peak_gpu:-}\t${calibration_peak_cpu:-}\t${SIFT_CALIBRATION_BATCH_SIZE}\t\t${train_ec}" >> "${METRICS_FILE}"
            fi
        fi
    else
        if [ ! -f "${ckpt}/config.json" ]; then
            echo "[mmlu] ${method} checkpoint not found for eval-only mode: ${ckpt}" | tee -a "${log_dir}/train.log"
            train_ec=1
        else
            echo "[mmlu] Skipping ${method} training; using checkpoint ${ckpt}" | tee -a "${log_dir}/train.log"
        fi
    fi

    if [ "${train_ec}" -ne 0 ]; then
        echo "| ${method} | train_failed:${train_ec} | | |" >> "${RESULTS_FILE}"
        echo "[mmlu] ${method} train failed with exit=${train_ec}"
        continue
    fi

    if [ "${method}" = "sift" ] && [[ "${SIFT_CALIBRATION_ONLY,,}" =~ ^(1|true|yes)$ ]]; then
        echo "[mmlu] ${method} calibration-only mode complete; skipping evaluation because no train checkpoint is saved." | tee -a "${log_dir}/train.log"
        continue
    fi

    if [ "${RUN_EVAL}" = "true" ]; then
        echo "============================================"
        echo "[mmlu] Evaluating ${method}"
        echo "============================================"
        eval_start="$(date +%s)"
        eval_peak_file="${log_dir}/eval_peak_memory_mb.txt"
        eval_stop_file="${log_dir}/eval_peak_memory.stop"
        start_gpu_monitor "${eval_peak_file}" "${eval_stop_file}" "${log_dir}/eval.log" "eval" "${method}" "${eval_start}" "$$"

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
        echo -e "${method}\teval\t${eval_elapsed}\t${eval_peak_mb}\t\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${eval_ec}" >> "${METRICS_FILE}"
        echo "[mmlu] ${method} eval exit=${eval_ec}"
        if [ "${eval_ec}" -ne 0 ]; then
            echo "| ${method} | eval_failed:${eval_ec} | | |" >> "${RESULTS_FILE}"
        fi
    fi
done

echo "Done. Results: ${RESULTS_FILE}"
echo "Metrics: ${METRICS_FILE}"
