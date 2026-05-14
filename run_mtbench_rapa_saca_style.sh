#!/usr/bin/env bash
# Standalone RAPA instruction-tuning + MT-Bench judge path.
# Uses SACA-style finetune behavior and PEFT merge_and_unload, without LMFlow merge wrappers.
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_PATH="${SPARSE_FT_ROOT}/$(basename "${BASH_SOURCE[0]}")"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

NUM_GPUS="${NUM_GPUS:-1}"
SLURM_TIME="${SLURM_TIME:-auto}"
maybe_reexec_with_srun "${SCRIPT_PATH}" "$@"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_BASE="${RAPA_HOME:-${LLM_ROOT}/rapa}"
SACA_BASE="${SACA_HOME:-${LLM_ROOT}/saca}"
RAPA_PEFT_DIR="${PEFT_DIR:-${RAPA_BASE}/peft}"
SACA_INSTRUCTION_TUNING_DIR="${SACA_INSTRUCTION_TUNING_DIR:-${SACA_BASE}/instruction-tuning}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-${SPARSE_FT_ROOT}/instruction_tuning/finetune_rapa_saca_style.py}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
DATASET="${DATASET:-oasst1}"
METHOD="${METHOD:-rapa}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_BASE}/rapa_saca_style_r602_mistral7b_mtbench_single_gpu}"
RESULTS="${RESULTS:-${RESULT_ROOT}/results.md}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"
RUN_REPORT_TSV="${RUN_REPORT_TSV:-${RESULT_ROOT}/run_report.tsv}"
RUN_REPORT_MD="${RUN_REPORT_MD:-${RESULT_ROOT}/run_report.md}"

LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-5e-5 1e-4}}"
RAPA_RANK="${RAPA_RANK:-602}"
RAPA_ALPHA="${RAPA_ALPHA:-1}"
RAPA_TARGET_MODULES="${RAPA_TARGET_MODULES:-no_head}"
RAPA_TRAIN_CONDA_ENV="${RAPA_TRAIN_CONDA_ENV:-saca}"
RAPA_RUN_PROJECT="${RAPA_RUN_PROJECT:-oasst1_mtbench_rapa_saca_style}"
RAPA_OFFLINE="${RAPA_OFFLINE:-true}"
RAPA_QUANTIZE="${RAPA_QUANTIZE:-false}"
RAPA_SAVE_FULL_MODEL="${RAPA_SAVE_FULL_MODEL:-true}"
RAPA_TRUST_REMOTE_CODE="${RAPA_TRUST_REMOTE_CODE:-false}"
RAPA_WANDB_DISABLED="${RAPA_WANDB_DISABLED:-false}"
RAPA_WANDB_MODE="${RAPA_WANDB_MODE:-offline}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
RAPA_WARMUP_RATIO="${RAPA_WARMUP_RATIO:-0.1}"
RAPA_SOURCE_MAX_LEN="${RAPA_SOURCE_MAX_LEN:-768}"
RAPA_TARGET_MAX_LEN="${RAPA_TARGET_MAX_LEN:-256}"
RAPA_DATALOADER_NUM_WORKERS="${RAPA_DATALOADER_NUM_WORKERS:-4}"
RAPA_TRAIN_ON_SOURCE="${RAPA_TRAIN_ON_SOURCE:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RESET_RESULTS="${RESET_RESULTS:-false}"
RUN_TAG="${RUN_TAG:-}"

DS_CONFIG="${DS_CONFIG:-${SPARSE_FT_ROOT}/instruction_tuning/ds_config_zero0_no_offload.json}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini-2024-07-18}"
LLM_JUDGE_DIR="${LLM_JUDGE_DIR:-${LLM_ROOT}/FastChat_rebuttal/fastchat/llm_judge}"
LLM_JUDGE_SCRIPT="${LLM_JUDGE_SCRIPT:-${LLM_JUDGE_DIR}/0_sparse_ft.sh}"
LLM_JUDGE_GPU="${LLM_JUDGE_GPU:-${GPU_INDEX:-0}}"
LLM_JUDGE_CONDA_ENV="${LLM_JUDGE_CONDA_ENV:-llmjudge_iclr2026}"
EVAL_SUMMARY_FILE="${EVAL_SUMMARY_FILE:-${RESULT_ROOT}/mtbench_judge_summary.tsv}"

GPU_INDEX="${GPU_INDEX:-0}"
if [ "${NUM_GPUS}" = "2" ]; then
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1}"
else
    DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:${GPU_INDEX}}"
fi

activate_sparse_ft_conda
configure_cuda_env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
export HF_HOME="${HF_HOME:-${RAPA_BASE}/hf_cache}"
export HF_TOKEN="${HF_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export RAPA_HOME="${RAPA_BASE}"
export PEFT_DIR="${RAPA_PEFT_DIR}"
export SACA_INSTRUCTION_TUNING_DIR
export PYTHONPATH="${RAPA_PEFT_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"

rapa_conda_run() {
    if [ -n "${RAPA_TRAIN_CONDA_ENV}" ]; then
        WANDB_DISABLED="${RAPA_WANDB_DISABLED}" WANDB_MODE="${RAPA_WANDB_MODE}" \
            conda run --no-capture-output -n "${RAPA_TRAIN_CONDA_ENV}" "$@"
    else
        WANDB_DISABLED="${RAPA_WANDB_DISABLED}" WANDB_MODE="${RAPA_WANDB_MODE}" "$@"
    fi
}

lr_label() {
    local lr="$1"
    lr="${lr//./p}"
    lr="${lr//+/_}"
    lr="${lr//-/_}"
    printf '%s' "${lr}"
}

sanitize_tag() {
    local value="$1"
    value="${value// /_}"
    value="${value//./p}"
    value="${value//+/_}"
    value="${value//-/_}"
    value="${value//,/_}"
    value="${value//:/_}"
    printf '%s' "${value}"
}

write_markdown_report() {
    {
        echo "# MT-Bench RAPA SACA-Style Instruction-Tuning Report"
        echo ""
        echo "- result_root: ${RESULT_ROOT}"
        echo "- judge_summary: ${EVAL_SUMMARY_FILE}"
        echo "- generated_at: $(date --iso-8601=seconds)"
        echo ""
        echo "| run_id | method | lr | tag | rank | alpha | targets | train_exit | eval_exit | turn1 | turn2 | avg | checkpoint |"
        echo "|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|"
        awk -F'\t' 'NR > 1 {
            printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, ($4 == "" ? "-" : $4), $5, $6, $7, $10, $12, ($13 == "" ? "-" : $13), ($14 == "" ? "-" : $14), ($15 == "" ? "-" : $15), $16)
        }' "${RUN_REPORT_TSV}"
    } > "${RUN_REPORT_MD}"
}

append_run_report() {
    local run_id="$1"
    local method="$2"
    local lr="$3"
    local checkpoint="$4"
    local train_elapsed="$5"
    local train_exit="$6"
    local eval_elapsed="$7"
    local eval_exit="$8"
    local turn1="$9"
    local turn2="${10}"
    local avg="${11}"

    local tmp="${RUN_REPORT_TSV}.tmp"
    awk -F'\t' -v run_id="${run_id}" 'NR == 1 || $1 != run_id' "${RUN_REPORT_TSV}" > "${tmp}"
    mv "${tmp}" "${RUN_REPORT_TSV}"
    echo -e "${run_id}\t${method}\t${lr}\t${RUN_TAG}\t${RAPA_RANK}\t${RAPA_ALPHA}\t${RAPA_TARGET_MODULES}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${train_exit}\t${train_elapsed}\t${eval_exit}\t${turn1}\t${turn2}\t${avg}\t${checkpoint}" >> "${RUN_REPORT_TSV}"
    write_markdown_report
}

mkdir -p "${RESULT_ROOT}/checkpoints" "${RESULT_ROOT}/logs"
if [ ! -f "${FINETUNE_SCRIPT}" ]; then
    echo "Finetune script not found: ${FINETUNE_SCRIPT}" >&2
    exit 1
fi
if [ ! -f "${DS_CONFIG}" ]; then
    echo "DeepSpeed config not found: ${DS_CONFIG}" >&2
    exit 1
fi
if [ "${RUN_EVAL}" = "true" ] && [ ! -x "${LLM_JUDGE_SCRIPT}" ]; then
    echo "LLM judge script not found or not executable: ${LLM_JUDGE_SCRIPT}" >&2
    exit 1
fi

if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RESULTS}" ]; then
    {
        echo "# MT-Bench RAPA SACA-Style Instruction-Tuning Results"
        echo ""
        echo "- model: ${MODEL}"
        echo "- dataset: ${DATASET}"
        echo "- method: ${METHOD}"
        echo "- learning_rates: ${LEARNING_RATES}"
        echo "- rank: ${RAPA_RANK}"
        echo "- alpha: ${RAPA_ALPHA}"
        echo "- target_modules: ${RAPA_TARGET_MODULES}"
        echo "- save_full_model: ${RAPA_SAVE_FULL_MODEL}"
        echo "- train_conda_env: ${RAPA_TRAIN_CONDA_ENV:-current}"
        echo "- finetune_script: ${FINETUNE_SCRIPT}"
        echo "- deepspeed_config: ${DS_CONFIG}"
        echo "- eval_script: ${LLM_JUDGE_SCRIPT}"
        echo "- judge_model: ${JUDGE_MODEL}"
        echo ""
    } > "${RESULTS}"
fi
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${METRICS_FILE}" ]; then
    echo -e "method\tlearning_rate\tphase\telapsed_seconds\tbatch_size\tgradient_accumulation_steps\toutput_dir\texit_code" > "${METRICS_FILE}"
fi
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RUN_REPORT_TSV}" ]; then
    echo -e "run_id\tmethod\tlearning_rate\trun_tag\trank\talpha\ttarget_modules\tbatch_size\tgradient_accumulation_steps\ttrain_exit\ttrain_elapsed_seconds\teval_exit\tturn1\tturn2\tavg\tcheckpoint" > "${RUN_REPORT_TSV}"
fi

for LR in ${LEARNING_RATES}; do
    LR_LABEL="$(lr_label "${LR}")"
    RUN_SUFFIX=""
    if [ -n "${RUN_TAG}" ]; then
        RUN_SUFFIX="_$(sanitize_tag "${RUN_TAG}")"
    fi
    RUN_ID="mtbench_${METHOD}_lr${LR_LABEL}${RUN_SUFFIX}"
    CKPT="${RESULT_ROOT}/checkpoints/${RUN_ID}"
    LOG="${RESULT_ROOT}/logs/${RUN_ID}"
    mkdir -p "${CKPT}" "${LOG}"

    TRAIN_EXIT=0
    TRAIN_ELAPSED=""
    if [ "${RUN_TRAIN}" = "true" ]; then
        echo "========================================"
        echo "[MT-Bench] SACA-style RAPA training: lr=${LR}, rank=${RAPA_RANK}, alpha=${RAPA_ALPHA}"
        echo "========================================"
        PORT=$((29600 + RANDOM % 1000))
        TRAIN_START="$(date +%s)"
        TRAIN_ARGS=()
        [ "${RAPA_OFFLINE}" = "true" ] && TRAIN_ARGS+=(--offline)
        [ "${RAPA_QUANTIZE}" = "true" ] && TRAIN_ARGS+=(--quantize)
        [ "${RAPA_SAVE_FULL_MODEL}" = "true" ] && TRAIN_ARGS+=(--save_full_model)
        [ "${RAPA_TRUST_REMOTE_CODE}" = "true" ] && TRAIN_ARGS+=(--trust_remote_code)
        [ "${RAPA_TRAIN_ON_SOURCE}" = "true" ] && TRAIN_ARGS+=(--train_on_source)

        set +e
        (
            cd "${SPARSE_FT_ROOT}"
            rapa_conda_run deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port="${PORT}" \
                "${FINETUNE_SCRIPT}" \
                --custom_mode "${METHOD}" \
                --lr "${LR}" \
                --lora_r "${RAPA_RANK}" \
                --lora_alpha "${RAPA_ALPHA}" \
                --train_bs "${BATCH_SIZE}" \
                --accumulation_steps "${GRAD_ACCUM}" \
                --model "${MODEL}" \
                --seed 42 \
                --logging_steps 1 \
                --target_modules "${RAPA_TARGET_MODULES}" \
                --metrics_enabled 0 \
                --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
                --warmup_ratio "${RAPA_WARMUP_RATIO}" \
                --run_project "${RAPA_RUN_PROJECT}" \
                --run_name "${RUN_ID}" \
                --run_id "${RUN_ID}" \
                --save_dir "${CKPT}" \
                --training_output_dir "${LOG}/training_output" \
                --deepspeed "${DS_CONFIG}" \
                --task instruct \
                --dataset "${DATASET}" \
                --epochs "${EPOCHS}" \
                --source_max_len "${RAPA_SOURCE_MAX_LEN}" \
                --target_max_len "${RAPA_TARGET_MAX_LEN}" \
                --dataloader_num_workers "${RAPA_DATALOADER_NUM_WORKERS}" \
                "${TRAIN_ARGS[@]}"
        ) 2>&1 | tee "${LOG}/train.log"
        TRAIN_EXIT=${PIPESTATUS[0]}
        set -e
        TRAIN_ELAPSED=$(( $(date +%s) - TRAIN_START ))
        echo -e "${METHOD}\t${LR}\ttrain\t${TRAIN_ELAPSED}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${CKPT}\t${TRAIN_EXIT}" >> "${METRICS_FILE}"
    elif [ ! -f "${CKPT}/config.json" ]; then
        echo "[MT-Bench] checkpoint not found for eval-only mode: ${CKPT}" | tee -a "${LOG}/train.log"
        TRAIN_EXIT=1
    fi

    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "[MT-Bench] ${METHOD}, lr=${LR} training FAILED (exit ${TRAIN_EXIT})"
        append_run_report "${RUN_ID}" "${METHOD}" "${LR}" "${CKPT}" "${TRAIN_ELAPSED}" "${TRAIN_EXIT}" "" "skipped" "" "" ""
        continue
    fi
    echo "[MT-Bench] ${METHOD}, lr=${LR} training DONE"

    EVAL_EXIT=""
    EVAL_ELAPSED=""
    TURN1=""
    TURN2=""
    AVG=""
    if [ "${RUN_EVAL}" = "true" ]; then
        echo "[MT-Bench] Evaluating with FastChat llm_judge: method=${METHOD}, lr=${LR}, run_tag=${RUN_TAG:-none}"
        EVAL_START="$(date +%s)"
        set +e
        RESULT_ROOT="${RESULT_ROOT}" \
        CONDA_ENV="${LLM_JUDGE_CONDA_ENV}" \
        JUDGE_MODEL="${JUDGE_MODEL}" \
        EVAL_SUMMARY_FILE="${EVAL_SUMMARY_FILE}" \
        RUN_TAG="${RUN_TAG}" \
        SKIP_MISSING=false \
        bash "${LLM_JUDGE_SCRIPT}" "${METHOD}" "${LLM_JUDGE_GPU}" "${LR}" "${RESULT_ROOT}" "${RUN_TAG}" \
            2>&1 | tee "${LOG}/eval.log"
        EVAL_EXIT=${PIPESTATUS[0]}
        set -e
        EVAL_ELAPSED=$(( $(date +%s) - EVAL_START ))
        echo -e "${METHOD}\t${LR}\tjudge\t${EVAL_ELAPSED}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${CKPT}\t${EVAL_EXIT}" >> "${METRICS_FILE}"
        if [ "${EVAL_EXIT}" -eq 0 ] && [ -f "${EVAL_SUMMARY_FILE}" ]; then
            read -r TURN1 TURN2 AVG <<< "$(awk -F'\t' -v run_id="${RUN_ID}" '$2 == run_id { turn1=$8; turn2=$9; avg=$10 } END { print turn1, turn2, avg }' "${EVAL_SUMMARY_FILE}")"
        fi
        echo "[MT-Bench] ${METHOD}, lr=${LR} llm_judge exit=${EVAL_EXIT}, avg=${AVG:-NA}"
    fi

    append_run_report "${RUN_ID}" "${METHOD}" "${LR}" "${CKPT}" "${TRAIN_ELAPSED}" "${TRAIN_EXIT}" "${EVAL_ELAPSED}" "${EVAL_EXIT}" "${TURN1}" "${TURN2}" "${AVG}"
done

echo "========================================"
echo "[MT-Bench] ALL RUNS COMPLETE"
echo "Results: ${RESULTS}"
echo "Run report: ${RUN_REPORT_MD}"
echo "Judge summary: ${EVAL_SUMMARY_FILE}"
echo "========================================"
