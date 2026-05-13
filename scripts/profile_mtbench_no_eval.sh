#!/usr/bin/env bash
set -euo pipefail

# MT-Bench / instruction-tuning profiling wrapper.
# Matches run_mtbench.sh training defaults, but runs only a short no-save profile
# and records memory/time metrics without evaluation.

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_BASE="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_BASE}/LMFlow}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"

DATA="${DATA:-${RAPA_BASE}/data/oasst1_lmflow.json}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_BASE}/profile_mtbench_no_eval}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/logs}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"
METHODS="${METHODS:-sift spiel smt s2ft ltsft}"

NUM_GPUS="${NUM_GPUS:-1}"
GPU_INDEX="${GPU_INDEX:-0}"
if [ "${NUM_GPUS}" = "2" ]; then
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1}"
else
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:${GPU_INDEX}}"
fi
MONITOR_GPU="${MONITOR_GPU:-${GPU_INDEX}}"

PROFILE_STEPS="${PROFILE_STEPS:-10}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
TARGET_PARAMS="${TARGET_PARAMS:-170000000}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-2}"
SYNC_LMFLOW="${SYNC_LMFLOW:-auto}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export HF_HOME="${HF_HOME:-${RAPA_BASE}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HF_HOME}/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HF_HOME}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}/xdg}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
export RAPA_HOME="${RAPA_BASE}"
export SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA_BASE}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"
export RAPA_SKIP_SAVE="${RAPA_SKIP_SAVE:-true}"
export RESUME_TRAINING="${RESUME_TRAINING:-false}"
export SIFT_USE_GRADIENT_CALIBRATION="${SIFT_USE_GRADIENT_CALIBRATION:-true}"
export SPIEL_DELTA_DTYPE="${SPIEL_DELTA_DTYPE:-float32}"
export SPIEL_SELECTION_ALGORITHM="${SPIEL_SELECTION_ALGORITHM:-rigl}"
export SPIEL_RESELECTION_STEPS="${SPIEL_RESELECTION_STEPS:-20}"
export SPIEL_SELECTION_ACCUMULATION_STEPS="${SPIEL_SELECTION_ACCUMULATION_STEPS:-5}"
export SPIEL_RESELECTION_RATE_POLICY="${SPIEL_RESELECTION_RATE_POLICY:-linear}"
export SPIEL_INITIAL_RESELECTION_RATE="${SPIEL_INITIAL_RESELECTION_RATE:-0.2}"
export SPIEL_TARGET_MODULES="${SPIEL_TARGET_MODULES:-q_proj,o_proj,v_proj,k_proj,gate_proj,up_proj,down_proj}"
export SPIEL_STRIP_DS_OPTIMIZER="${SPIEL_STRIP_DS_OPTIMIZER:-true}"
export SIFT_CALIBRATION_STEPS="${SIFT_CALIBRATION_STEPS:-1}"
export SIFT_CALIBRATION_BATCH_SIZE="${SIFT_CALIBRATION_BATCH_SIZE:-1}"
export SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-100}"
export SMT_CALIBRATION_BATCH_SIZE="${SMT_CALIBRATION_BATCH_SIZE:-1}"
export S2FT_CALIBRATION_STEPS="${S2FT_CALIBRATION_STEPS:-100}"
export S2FT_CALIBRATION_BATCH_SIZE="${S2FT_CALIBRATION_BATCH_SIZE:-1}"
export LTSFT_MASK_SEARCH_STEPS="${LTSFT_MASK_SEARCH_STEPS:-100}"
export LTSFT_N_FT_ITERATIONS="${LTSFT_N_FT_ITERATIONS:-1}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${HF_HOME}"

activate_sparse_ft_conda
configure_cuda_env

if [ ! -f "${DATA}" ]; then
    echo "Dataset not found: ${DATA}" >&2
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set." >&2
    exit 1
fi

sync_lmflow_sources() {
    local target_dir="${LMFLOW_DIR}/src/lmflow/pipeline/rapa"
    local target_config_dir="${LMFLOW_DIR}/configs/rapa"
    local avail_kb

    if [ "${SYNC_LMFLOW}" = "false" ]; then
        echo "[sync] skipping LMFlow sync (SYNC_LMFLOW=false)"
        return 0
    fi

    mkdir -p "${target_dir}" "${target_config_dir}"
    avail_kb="$(df -Pk "${LMFLOW_DIR}" | awk 'NR == 2 { print int($4) }')"
    if [ "${SYNC_LMFLOW}" = "auto" ] && [ "${avail_kb}" -lt 65536 ]; then
        echo "[sync] skipping LMFlow sync: only ${avail_kb}KB available"
        return 0
    fi

    cp -r "${SPARSE_FT_ROOT}/pipeline/." "${target_dir}/"
    cp -r "${SPARSE_FT_ROOT}/configs/." "${target_config_dir}/"
}

sync_lmflow_sources

if [ ! -f "${METRICS_FILE}" ]; then
    printf 'timestamp\tmethod\tinclude\tmonitor_gpu\texit_code\telapsed_sec\tprofile_steps\testimated_total_steps\ttrain_runtime_sec\testimated_train_sec\testimated_total_wall_sec\tweight_selection_sec\tcalibration_sec\tbaseline_gpu_mb\tpeak_gpu_mb\tpeak_gpu_delta_mb\tpeak_cpu_rss_mb\tpeak_total_mb\tlog_file\toutput_dir\n' > "${METRICS_FILE}"
fi

dataset_size="$(
    python - <<PY
import json
with open("${DATA}") as f:
    raw = json.load(f)
print(len(raw.get("instances", [])))
PY
)"
effective_batch=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
estimated_total_steps=$(( (dataset_size + effective_batch - 1) / effective_batch * EPOCHS ))

descendant_pids() {
    local root_pid="$1"
    local pending="${root_pid}"
    local all="${root_pid}"
    local next pid children
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

gpu_used_mb() {
    local gpu="$1"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
        | awk 'NR == 1 { gsub(/ /, ""); print int($1); found=1 } END { if (!found) print 0 }'
}

start_monitor() {
    local root_pid="$1"
    local gpu="$2"
    local out_file="$3"
    local marker="${4:-}"
    : > "${out_file}"
    (
        while kill -0 "${root_pid}" >/dev/null 2>&1; do
            root_cpu="$(job_cpu_rss_mb "${root_pid}")"
            marker_cpu=0
            if [ -n "${marker}" ]; then
                marker_cpu="$(marker_cpu_rss_mb "${marker}")"
            fi
            if [ "${marker_cpu}" -gt "${root_cpu}" ]; then
                root_cpu="${marker_cpu}"
            fi
            printf '%s,%s,%s\n' "$(date +%s)" "$(gpu_used_mb "${gpu}")" "${root_cpu}" >> "${out_file}"
            sleep "${GPU_MONITOR_INTERVAL}"
        done
        printf '%s,%s,%s\n' "$(date +%s)" "$(gpu_used_mb "${gpu}")" "$(job_cpu_rss_mb "${root_pid}")" >> "${out_file}"
    ) &
    echo "$!"
}

metric_from_log() {
    local log_file="$1"
    local pattern="$2"
    grep -E "${pattern}" "${log_file}" 2>/dev/null | tail -n 1 | awk -F'=' '{gsub(/[^0-9.]/, "", $2); print $2}'
}

train_runtime_from_log() {
    local log_file="$1"
    grep -oE "'train_runtime': [0-9.]+" "${log_file}" 2>/dev/null | tail -n 1 | awk '{print $2}'
}

append_metrics() {
    local method="$1" exit_code="$2" elapsed="$3" log_file="$4" output_dir="$5" monitor_file="$6" baseline_gpu="$7"
    local train_runtime weight_selection calibration peak_gpu peak_gpu_delta peak_cpu peak_total estimated_train estimated_wall

    train_runtime="$(train_runtime_from_log "${log_file}")"
    weight_selection="$(metric_from_log "${log_file}" "weight_selection_seconds" || true)"
    calibration="$(metric_from_log "${log_file}" "calibration_seconds" || true)"
    peak_gpu="$(awk -F',' 'BEGIN{m=0} {if ($2+0>m)m=$2+0} END{print m+0}' "${monitor_file}" 2>/dev/null || echo 0)"
    peak_gpu_delta="$(awk -v p="${peak_gpu}" -v b="${baseline_gpu}" 'BEGIN{d=p-b; if (d<0) d=0; printf "%.0f", d}')"
    peak_cpu="$(awk -F',' 'BEGIN{m=0} {if ($3+0>m)m=$3+0} END{print m+0}' "${monitor_file}" 2>/dev/null || echo 0)"
    peak_total="$(awk -v g="${peak_gpu}" -v c="${peak_cpu}" 'BEGIN{printf "%.0f", g + c}')"

    if [ -n "${train_runtime}" ] && awk "BEGIN{exit !(${PROFILE_STEPS} > 0)}"; then
        estimated_train="$(awk -v rt="${train_runtime}" -v ps="${PROFILE_STEPS}" -v ts="${estimated_total_steps}" 'BEGIN{printf "%.2f", rt / ps * ts}')"
        estimated_wall="$(awk -v elapsed="${elapsed}" -v rt="${train_runtime}" -v et="${estimated_train}" 'BEGIN{printf "%.2f", elapsed - rt + et}')"
    else
        estimated_train=""
        estimated_wall=""
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TIMESTAMP}" "${method}" "${DEEPSPEED_INCLUDE}" "${MONITOR_GPU}" "${exit_code}" "${elapsed}" \
        "${PROFILE_STEPS}" "${estimated_total_steps}" "${train_runtime}" "${estimated_train}" \
        "${estimated_wall}" "${weight_selection}" "${calibration}" "${baseline_gpu}" "${peak_gpu}" \
        "${peak_gpu_delta}" "${peak_cpu}" "${peak_total}" "${log_file}" "${output_dir}" >> "${METRICS_FILE}"
}

run_method() {
    local method="$1"
    local run_name output_dir log_dir log_file monitor_file port start elapsed exit_code monitor_pid baseline_gpu

    run_name="${method}_${TIMESTAMP}"
    output_dir="${RESULT_ROOT}/checkpoints/${run_name}"
    log_dir="${LOG_ROOT}/${run_name}"
    log_file="${log_dir}/train.log"
    monitor_file="${log_dir}/monitor.csv"
    mkdir -p "${output_dir}" "${log_dir}"

    port=$((29600 + RANDOM % 1000))
    baseline_gpu="$(gpu_used_mb "${MONITOR_GPU}")"
    start="$(date +%s)"

    (
        cd "${LMFLOW_DIR}"
        deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port="${port}" \
            "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py" \
            --method "${method}" \
            --model_name_or_path "${MODEL}" \
            --dataset_path "${DATA}" \
            --output_dir "${output_dir}" \
            --num_train_epochs "${EPOCHS}" \
            --max_steps "${PROFILE_STEPS}" \
            --per_device_train_batch_size "${BATCH_SIZE}" \
            --gradient_accumulation_steps "${GRAD_ACCUM}" \
            --learning_rate "${LEARNING_RATE}" \
            --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
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
            --hf_token "${HF_TOKEN}" \
            --seed 42
    ) > "${log_file}" 2>&1 &
    train_pid="$!"
    monitor_pid="$(start_monitor "${train_pid}" "${MONITOR_GPU}" "${monitor_file}" "${output_dir}")"

    set +e
    wait "${train_pid}"
    exit_code=$?
    set -e
    wait "${monitor_pid}" 2>/dev/null || true

    elapsed=$(( $(date +%s) - start ))
    append_metrics "${method}" "${exit_code}" "${elapsed}" "${log_file}" "${output_dir}" "${monitor_file}" "${baseline_gpu}"
    echo "[profile-mtbench] method=${method} include=${DEEPSPEED_INCLUDE} exit=${exit_code} elapsed=${elapsed}s log=${log_file}"
}

echo "[profile-mtbench] dataset_size=${dataset_size} estimated_total_steps=${estimated_total_steps}"
echo "[profile-mtbench] model=${MODEL}"
echo "[profile-mtbench] data=${DATA}"
echo "[profile-mtbench] methods=${METHODS}"
echo "[profile-mtbench] include=${DEEPSPEED_INCLUDE} monitor_gpu=${MONITOR_GPU}"

for method in ${METHODS}; do
    run_method "${method}"
done

echo "[profile-mtbench] metrics: ${METRICS_FILE}"
