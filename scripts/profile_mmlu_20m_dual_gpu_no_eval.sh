#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DATASET="${DATASET:-${RAPA_HOME}/OwLore_Dataset/mmlu/mmlu.json}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profile_mmlu_20m_dual_gpu_no_eval}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/logs}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"
ATTEMPTS_FILE="${ATTEMPTS_FILE:-${RESULT_ROOT}/attempts.tsv}"
GPU0_METHODS="${GPU0_METHODS-sift smt ltsft}"
GPU1_METHODS="${GPU1_METHODS-spiel s2ft}"
CPU_OFFLOAD_METHODS="${CPU_OFFLOAD_METHODS:-ltsft}"
ENABLE_OFFLOAD_RETRY="${ENABLE_OFFLOAD_RETRY:-true}"
OFFLOAD_RETRY_ON_ANY_FAILURE="${OFFLOAD_RETRY_ON_ANY_FAILURE:-false}"
PROFILE_STEPS="${PROFILE_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TARGET_PARAMS="${TARGET_PARAMS:-20000000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-2}"
SYNC_LMFLOW="${SYNC_LMFLOW:-auto}"
BASE_DS_CONFIG="${BASE_DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"
OFFLOAD_DS_CONFIG="${OFFLOAD_DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero2_offload.json}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export HF_HOME="${HF_HOME:-${RAPA_HOME}/hf_cache}"
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
export RAPA_HOME SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA_HOME}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export RAPA_SKIP_SAVE="${RAPA_SKIP_SAVE:-true}"
export RESUME_TRAINING="${RESUME_TRAINING:-false}"
export SIFT_USE_GRADIENT_CALIBRATION="${SIFT_USE_GRADIENT_CALIBRATION:-true}"
export SIFT_CALIBRATION_STEPS="${SIFT_CALIBRATION_STEPS:-1}"
export SIFT_CALIBRATION_BATCH_SIZE="${SIFT_CALIBRATION_BATCH_SIZE:-1}"
export SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-100}"
export SMT_CALIBRATION_BATCH_SIZE="${SMT_CALIBRATION_BATCH_SIZE:-1}"
export S2FT_CALIBRATION_STEPS="${S2FT_CALIBRATION_STEPS:-100}"
export S2FT_CALIBRATION_BATCH_SIZE="${S2FT_CALIBRATION_BATCH_SIZE:-1}"
export S2FT_SELECTION_METHOD="${S2FT_SELECTION_METHOD:-random}"
export S2FT_TARGET_PROJECTIONS="${S2FT_TARGET_PROJECTIONS:-v,o,u,d}"
export S2FT_V_RATIO="${S2FT_V_RATIO:-}"
export S2FT_O_RATIO="${S2FT_O_RATIO:-}"
export S2FT_U_RATIO="${S2FT_U_RATIO:-}"
export S2FT_D_RATIO="${S2FT_D_RATIO:-}"
export LTSFT_MASK_SEARCH_STEPS="${LTSFT_MASK_SEARCH_STEPS:-100}"
export LTSFT_N_FT_ITERATIONS="${LTSFT_N_FT_ITERATIONS:-1}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${HF_HOME}"

activate_sparse_ft_conda
configure_cuda_env

if [ ! -f "${DATASET}" ]; then
    echo "Dataset not found: ${DATASET}" >&2
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

printf 'timestamp\tmethod\tgpu\tfinal_offload_enabled\tretry_used\tfinal_exit_code\ttotal_elapsed_sec\tprimary_elapsed_sec\toffload_elapsed_sec\tprofile_steps\testimated_total_steps\ttrain_runtime_sec\testimated_train_sec\testimated_total_wall_sec\tweight_selection_sec\tcalibration_sec\tpeak_gpu_mb\tpeak_cpu_rss_mb\tpeak_total_mb\tbatch_size\tgrad_accum\tlearning_rate\tlog_file\toutput_dir\n' > "${METRICS_FILE}"
printf 'timestamp\tmethod\tgpu\tattempt\toffload_enabled\texit_code\telapsed_sec\tprofile_steps\testimated_total_steps\ttrain_runtime_sec\testimated_train_sec\testimated_total_wall_sec\tweight_selection_sec\tcalibration_sec\tpeak_gpu_mb\tpeak_cpu_rss_mb\tpeak_total_mb\tbatch_size\tgrad_accum\tlearning_rate\tlog_file\toutput_dir\n' > "${ATTEMPTS_FILE}"

dataset_size="$(
    python - <<PY
import json
with open("${DATASET}") as f:
    raw = json.load(f)
print(len(raw.get("instances", [])))
PY
)"
estimated_total_steps=$(( (dataset_size + BATCH_SIZE * GRAD_ACCUM - 1) / (BATCH_SIZE * GRAD_ACCUM) * EPOCHS ))

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
        root_cpu="$(job_cpu_rss_mb "${root_pid}")"
        marker_cpu=0
        if [ -n "${marker}" ]; then
            marker_cpu="$(marker_cpu_rss_mb "${marker}")"
        fi
        if [ "${marker_cpu}" -gt "${root_cpu}" ]; then
            root_cpu="${marker_cpu}"
        fi
        printf '%s,%s,%s\n' "$(date +%s)" "$(gpu_used_mb "${gpu}")" "${root_cpu}" >> "${out_file}"
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

method_uses_offload() {
    local method="$1"
    for item in ${CPU_OFFLOAD_METHODS}; do
        if [ "${item}" = "${method}" ]; then
            return 0
        fi
    done
    return 1
}

is_oom_failure() {
    local log_file="$1"
    grep -Eiq 'out of memory|CUDA error: out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|DeepSpeed.*OOM|OOM' "${log_file}" 2>/dev/null
}

append_attempt_metrics() {
    local method="$1" gpu="$2" attempt="$3" offload="$4" exit_code="$5" elapsed="$6" log_file="$7" output_dir="$8" monitor_file="$9"
    local train_runtime weight_selection calibration peak_gpu peak_cpu estimated_train estimated_wall internal_gpu internal_cpu peak_total

    train_runtime="$(train_runtime_from_log "${log_file}")"
    weight_selection="$(metric_from_log "${log_file}" "weight_selection_seconds" || true)"
    calibration="$(metric_from_log "${log_file}" "calibration_seconds" || true)"
    peak_gpu="$(awk -F',' 'BEGIN{m=0} {if ($2+0>m)m=$2+0} END{print m+0}' "${monitor_file}" 2>/dev/null || echo 0)"
    peak_cpu="$(awk -F',' 'BEGIN{m=0} {if ($3+0>m)m=$3+0} END{print m+0}' "${monitor_file}" 2>/dev/null || echo 0)"
    internal_gpu="$(metric_from_log "${log_file}" "calibration_peak_reserved_mb" || true)"
    internal_cpu="$(metric_from_log "${log_file}" "calibration_peak_cpu_rss_mb" || true)"
    if [ -n "${internal_gpu}" ] && awk -v a="${internal_gpu}" -v b="${peak_gpu}" 'BEGIN{exit !(a>b)}'; then
        peak_gpu="${internal_gpu}"
    fi
    if [ -n "${internal_cpu}" ] && awk -v a="${internal_cpu}" -v b="${peak_cpu}" 'BEGIN{exit !(a>b)}'; then
        peak_cpu="${internal_cpu}"
    fi
    peak_total="$(awk -v g="${peak_gpu}" -v c="${peak_cpu}" 'BEGIN{printf "%.0f", g + c}')"

    if [ -n "${train_runtime}" ] && awk "BEGIN{exit !(${PROFILE_STEPS} > 0)}"; then
        estimated_train="$(awk -v rt="${train_runtime}" -v ps="${PROFILE_STEPS}" -v ts="${estimated_total_steps}" 'BEGIN{printf "%.2f", rt / ps * ts}')"
        estimated_wall="$(awk -v elapsed="${elapsed}" -v rt="${train_runtime}" -v et="${estimated_train}" 'BEGIN{printf "%.2f", elapsed - rt + et}')"
    else
        estimated_train=""
        estimated_wall=""
    fi

    LAST_TRAIN_RUNTIME="${train_runtime}"
    LAST_WEIGHT_SELECTION="${weight_selection}"
    LAST_CALIBRATION="${calibration}"
    LAST_PEAK_GPU="${peak_gpu}"
    LAST_PEAK_CPU="${peak_cpu}"
    LAST_PEAK_TOTAL="${peak_total}"
    LAST_ESTIMATED_TRAIN="${estimated_train}"
    LAST_ESTIMATED_WALL="${estimated_wall}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TIMESTAMP}" "${method}" "${gpu}" "${attempt}" "${offload}" "${exit_code}" "${elapsed}" \
        "${PROFILE_STEPS}" "${estimated_total_steps}" "${train_runtime}" "${estimated_train}" \
        "${estimated_wall}" "${weight_selection}" "${calibration}" "${peak_gpu}" "${peak_cpu}" \
        "${peak_total}" "${BATCH_SIZE}" "${GRAD_ACCUM}" "${LEARNING_RATE}" "${log_file}" "${output_dir}" >> "${ATTEMPTS_FILE}"
}

append_method_summary() {
    local method="$1" gpu="$2" final_offload="$3" retry_used="$4" final_exit_code="$5" total_elapsed="$6"
    local primary_elapsed="$7" offload_elapsed="$8" train_runtime="$9" estimated_train="${10}"
    local estimated_wall="${11}" weight_selection="${12}" calibration="${13}" peak_gpu="${14}"
    local peak_cpu="${15}" peak_total="${16}" log_file="${17}" output_dir="${18}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TIMESTAMP}" "${method}" "${gpu}" "${final_offload}" "${retry_used}" "${final_exit_code}" \
        "${total_elapsed}" "${primary_elapsed}" "${offload_elapsed}" "${PROFILE_STEPS}" \
        "${estimated_total_steps}" "${train_runtime}" "${estimated_train}" "${estimated_wall}" \
        "${weight_selection}" "${calibration}" "${peak_gpu}" "${peak_cpu}" "${peak_total}" \
        "${BATCH_SIZE}" "${GRAD_ACCUM}" "${LEARNING_RATE}" "${log_file}" "${output_dir}" >> "${METRICS_FILE}"
}

run_one_attempt() {
    local method="$1" gpu="$2" offload="$3" attempt="$4"
    local run_name output_dir log_dir log_file monitor_file port start elapsed exit_code ds_config monitor_pid

    run_name="${method}_gpu${gpu}_${attempt}_${TIMESTAMP}"
    output_dir="${RESULT_ROOT}/checkpoints/${run_name}"
    log_dir="${LOG_ROOT}/${run_name}"
    log_file="${log_dir}/train.log"
    monitor_file="${log_dir}/monitor.csv"
    mkdir -p "${output_dir}" "${log_dir}"

    if [ "${offload}" = "true" ]; then
        ds_config="${OFFLOAD_DS_CONFIG}"
    else
        ds_config="${BASE_DS_CONFIG}"
    fi

    local s2ft_extra_args=()
    if [ "${method}" = "s2ft" ]; then
        if [ -n "${S2FT_SELECTION_METHOD}" ]; then
            s2ft_extra_args+=(--s2ft_selection_method "${S2FT_SELECTION_METHOD}")
        fi
        if [ -n "${S2FT_V_RATIO}" ] && [ "${S2FT_V_RATIO}" != "auto" ]; then
            s2ft_extra_args+=(--s2ft_v_ratio "${S2FT_V_RATIO}")
        fi
        if [ -n "${S2FT_O_RATIO}" ] && [ "${S2FT_O_RATIO}" != "auto" ]; then
            s2ft_extra_args+=(--s2ft_o_ratio "${S2FT_O_RATIO}")
        fi
        if [ -n "${S2FT_U_RATIO}" ] && [ "${S2FT_U_RATIO}" != "auto" ]; then
            s2ft_extra_args+=(--s2ft_u_ratio "${S2FT_U_RATIO}")
        fi
        if [ -n "${S2FT_D_RATIO}" ] && [ "${S2FT_D_RATIO}" != "auto" ]; then
            s2ft_extra_args+=(--s2ft_d_ratio "${S2FT_D_RATIO}")
        fi
    fi

    port=$((29500 + RANDOM % 1000))
    start="$(date +%s)"

    (
        cd "${LMFLOW_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export DS_CONFIG="${ds_config}"
        deepspeed --include=localhost:"${gpu}" --master_port="${port}" \
            "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py" \
            --method "${method}" \
            --model_name_or_path "${MODEL}" \
            --dataset_path "${DATASET}" \
            --output_dir "${output_dir}" \
            --num_train_epochs "${EPOCHS}" \
            --max_steps "${PROFILE_STEPS}" \
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
            "${s2ft_extra_args[@]}" \
            --ltsft_mask_search_steps "${LTSFT_MASK_SEARCH_STEPS}" \
            --ltsft_n_ft_iterations "${LTSFT_N_FT_ITERATIONS}" \
            --bf16 \
            --seed 42
    ) > "${log_file}" 2>&1 &
    train_pid="$!"
    monitor_pid="$(start_monitor "${train_pid}" "${gpu}" "${monitor_file}" "${output_dir}")"

    set +e
    wait "${train_pid}"
    exit_code=$?
    set -e
    wait "${monitor_pid}" 2>/dev/null || true

    elapsed=$(( $(date +%s) - start ))
    append_attempt_metrics "${method}" "${gpu}" "${attempt}" "${offload}" "${exit_code}" "${elapsed}" "${log_file}" "${output_dir}" "${monitor_file}"

    LAST_EXIT_CODE="${exit_code}"
    LAST_ELAPSED="${elapsed}"
    LAST_OFFLOAD="${offload}"
    LAST_LOG_FILE="${log_file}"
    LAST_OUTPUT_DIR="${output_dir}"
    LAST_OOM=false
    if is_oom_failure "${log_file}"; then
        LAST_OOM=true
    fi

    echo "[profile] method=${method} gpu=${gpu} offload=${offload} exit=${exit_code} elapsed=${elapsed}s log=${log_file}"
    return "${exit_code}"
}

run_method() {
    local method="$1" gpu="$2" use_offload=false
    local retry_used=false total_elapsed primary_elapsed=0 offload_elapsed=0
    local primary_peak_gpu=0 primary_peak_cpu=0 primary_peak_total=0
    local final_exit final_offload final_train_runtime final_estimated_train final_estimated_wall
    local final_weight_selection final_calibration final_peak_gpu final_peak_cpu final_peak_total
    local final_log_file final_output_dir
    if method_uses_offload "${method}"; then
        use_offload=true
    fi

    if run_one_attempt "${method}" "${gpu}" "${use_offload}" "primary"; then
        :
    else
        primary_peak_gpu="${LAST_PEAK_GPU}"
        primary_peak_cpu="${LAST_PEAK_CPU}"
        primary_peak_total="${LAST_PEAK_TOTAL}"
        if [ "${use_offload}" != "true" ] && [ "${ENABLE_OFFLOAD_RETRY}" = "true" ] && \
           { [ "${OFFLOAD_RETRY_ON_ANY_FAILURE}" = "true" ] || [ "${LAST_OOM}" = "true" ]; }; then
            echo "[profile] ${method} failed with likely OOM; retrying with CPU optimizer offload at the same batch_size=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}"
            primary_elapsed="${LAST_ELAPSED}"
            retry_used=true
            run_one_attempt "${method}" "${gpu}" "true" "offload_retry" || true
            offload_elapsed="${LAST_ELAPSED}"
        fi
    fi

    if [ "${retry_used}" != "true" ]; then
        primary_elapsed="${LAST_ELAPSED}"
        if [ "${LAST_OFFLOAD}" = "true" ]; then
            offload_elapsed="${LAST_ELAPSED}"
        fi
    fi
    if [ "${retry_used}" = "true" ]; then
        total_elapsed=$((primary_elapsed + offload_elapsed))
    else
        total_elapsed="${primary_elapsed}"
    fi
    final_exit="${LAST_EXIT_CODE}"
    final_offload="${LAST_OFFLOAD}"
    final_train_runtime="${LAST_TRAIN_RUNTIME}"
    final_estimated_train="${LAST_ESTIMATED_TRAIN}"
    final_estimated_wall="${LAST_ESTIMATED_WALL}"
    if [ "${retry_used}" = "true" ] && [ -n "${final_estimated_wall}" ]; then
        final_estimated_wall="$(awk -v p="${primary_elapsed}" -v e="${final_estimated_wall}" 'BEGIN{printf "%.2f", p + e}')"
    fi
    final_weight_selection="${LAST_WEIGHT_SELECTION}"
    final_calibration="${LAST_CALIBRATION}"
    final_peak_gpu="${LAST_PEAK_GPU}"
    final_peak_cpu="${LAST_PEAK_CPU}"
    final_peak_total="${LAST_PEAK_TOTAL}"
    if [ "${retry_used}" = "true" ]; then
        final_peak_gpu="$(awk -v a="${primary_peak_gpu}" -v b="${final_peak_gpu}" 'BEGIN{print (a>b ? a : b)}')"
        final_peak_cpu="$(awk -v a="${primary_peak_cpu}" -v b="${final_peak_cpu}" 'BEGIN{print (a>b ? a : b)}')"
        final_peak_total="$(awk -v a="${primary_peak_total}" -v b="${final_peak_total}" 'BEGIN{print (a>b ? a : b)}')"
    fi
    final_log_file="${LAST_LOG_FILE}"
    final_output_dir="${LAST_OUTPUT_DIR}"

    append_method_summary "${method}" "${gpu}" "${final_offload}" "${retry_used}" "${final_exit}" \
        "${total_elapsed}" "${primary_elapsed}" "${offload_elapsed}" "${final_train_runtime}" \
        "${final_estimated_train}" "${final_estimated_wall}" "${final_weight_selection}" \
        "${final_calibration}" "${final_peak_gpu}" "${final_peak_cpu}" "${final_peak_total}" \
        "${final_log_file}" "${final_output_dir}"

    return 0
}

worker() {
    local gpu="$1"
    shift
    local methods=("$@")
    for method in "${methods[@]}"; do
        run_method "${method}" "${gpu}"
    done
}

echo "[profile] dataset_size=${dataset_size} estimated_total_steps=${estimated_total_steps}"
echo "[profile] GPU0 methods: ${GPU0_METHODS}"
echo "[profile] GPU1 methods: ${GPU1_METHODS}"

worker 0 ${GPU0_METHODS} &
pid0=$!
worker 1 ${GPU1_METHODS} &
pid1=$!

wait "${pid0}"
wait "${pid1}"

echo "[profile] metrics: ${METRICS_FILE}"
