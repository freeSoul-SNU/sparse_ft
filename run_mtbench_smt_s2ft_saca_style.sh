#!/usr/bin/env bash
# Standalone SACA-style instruction tuning for SMT/S2FT + MT-Bench judge.
# This path uses local sparse-ft code only; it does not sync through LMFlow.
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
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-${SPARSE_FT_ROOT}/instruction_tuning/finetune_smt_s2ft_saca_style.py}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
DATASET="${DATASET:-oasst1}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_BASE}/smt_s2ft_saca_style_170m_mistral7b_mtbench_single_gpu}"
RESULTS="${RESULTS:-${RESULT_ROOT}/results.md}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"
RUN_REPORT_TSV="${RUN_REPORT_TSV:-${RESULT_ROOT}/run_report.tsv}"
RUN_REPORT_MD="${RUN_REPORT_MD:-${RESULT_ROOT}/run_report.md}"

METHODS="${METHODS:-smt s2ft}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-1e-4 5e-5}}"
TARGET_PARAMS="${TARGET_PARAMS:-170000000}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SOURCE_MAX_LEN="${SOURCE_MAX_LEN:-768}"
TARGET_MAX_LEN="${TARGET_MAX_LEN:-256}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-${SOURCE_MAX_LEN}}"
TRAIN_ON_SOURCE="${TRAIN_ON_SOURCE:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RESET_RESULTS="${RESET_RESULTS:-false}"
RUN_TAG="${RUN_TAG:-}"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-sparse-ft}"
REPORT_TO="${REPORT_TO:-none}"
SACA_STYLE_QUANTIZE="${SACA_STYLE_QUANTIZE:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-20}"
SMT_CALIBRATION_BATCH_SIZE="${SMT_CALIBRATION_BATCH_SIZE:-1}"
if [ -n "${SMT_TARGET_MODULES:-}" ]; then
    echo "[mtbench] Ignoring legacy SMT_TARGET_MODULES=${SMT_TARGET_MODULES}; use SMT_ATTENTION_TARGET_MODULES/SMT_MLP_TARGET_MODULES instead."
    unset SMT_TARGET_MODULES
fi
SMT_ATTENTION_TARGET_MODULES="${SMT_ATTENTION_TARGET_MODULES:-q_proj,k_proj,v_proj}"
SMT_MLP_TARGET_MODULES="${SMT_MLP_TARGET_MODULES:-gate_proj,up_proj,down_proj}"
SMT_BUDGET_ALLOCATION="${SMT_BUDGET_ALLOCATION:-attention_only}"
SMT_SELECTION_STRATEGY="${SMT_SELECTION_STRATEGY:-no_restriction}"
SMT_CALCULATION_STRATEGY="${SMT_CALCULATION_STRATEGY:-mean_abs}"

S2FT_CALIBRATION_STEPS="${S2FT_CALIBRATION_STEPS:-0}"
S2FT_CALIBRATION_BATCH_SIZE="${S2FT_CALIBRATION_BATCH_SIZE:-1}"
S2FT_SELECTION_METHOD="${S2FT_SELECTION_METHOD:-random}"
S2FT_RATIO_PRESET="${S2FT_RATIO_PRESET:-budget}"
S2FT_LAYER_ALLOCATION="${S2FT_LAYER_ALLOCATION:-uniform}"
S2FT_TARGET_PROJECTIONS="${S2FT_TARGET_PROJECTIONS:-d}"

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
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export RAPA_HOME="${RAPA_BASE}"
export SPARSE_FT_ROOT
export PYTHONPATH="${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export DS_CONFIG
export SMT_CALIBRATION_STEPS SMT_CALIBRATION_BATCH_SIZE
export SMT_ATTENTION_TARGET_MODULES SMT_MLP_TARGET_MODULES
export SMT_BUDGET_ALLOCATION SMT_SELECTION_STRATEGY SMT_CALCULATION_STRATEGY
export S2FT_CALIBRATION_STEPS S2FT_CALIBRATION_BATCH_SIZE
export S2FT_SELECTION_METHOD S2FT_RATIO_PRESET S2FT_LAYER_ALLOCATION S2FT_TARGET_PROJECTIONS

train_conda_run() {
    if [ -n "${TRAIN_CONDA_ENV}" ]; then
        conda run --no-capture-output -n "${TRAIN_CONDA_ENV}" "$@"
    else
        "$@"
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
        echo "# MT-Bench SMT/S2FT SACA-Style Run Report"
        echo ""
        echo "- result_root: ${RESULT_ROOT}"
        echo "- judge_summary: ${EVAL_SUMMARY_FILE}"
        echo "- generated_at: $(date --iso-8601=seconds)"
        echo ""
        echo "| run_id | method | lr | tag | settings | batch | grad_accum | seq | train_exit | eval_exit | turn1 | turn2 | avg | checkpoint |"
        echo "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        awk -F'\t' 'NR > 1 {
            settings = "target=" $5 ", smt_cal=" $12 ", smt_budget=" $16 ", smt_sel=" $17 ", s2ft_cal=" $19 ", s2ft_sel=" $21 ", s2ft_target=" $24 ", bf16=" $27
            printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, ($4 == "" ? "-" : $4), settings, $6, $7, $8, $29, $31, ($32 == "" ? "-" : $32), ($33 == "" ? "-" : $33), ($34 == "" ? "-" : $34), $35)
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
    echo -e "${run_id}\t${method}\t${lr}\t${RUN_TAG}\t${TARGET_PARAMS}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${MAX_SEQ_LENGTH}\t${LR_SCHEDULER_TYPE}\t${WARMUP_STEPS}\t${WARMUP_RATIO}\t${SMT_CALIBRATION_STEPS}\t${SMT_CALIBRATION_BATCH_SIZE}\t${SMT_ATTENTION_TARGET_MODULES}\t${SMT_MLP_TARGET_MODULES}\t${SMT_BUDGET_ALLOCATION}\t${SMT_SELECTION_STRATEGY}\t${SMT_CALCULATION_STRATEGY}\t${S2FT_CALIBRATION_STEPS}\t${S2FT_CALIBRATION_BATCH_SIZE}\t${S2FT_SELECTION_METHOD}\t${S2FT_RATIO_PRESET}\t${S2FT_LAYER_ALLOCATION}\t${S2FT_TARGET_PROJECTIONS}\t${SOURCE_MAX_LEN}\t${TARGET_MAX_LEN}\t${SACA_STYLE_QUANTIZE}\t${train_elapsed}\t${train_exit}\t${eval_elapsed}\t${eval_exit}\t${turn1}\t${turn2}\t${avg}\t${checkpoint}" >> "${RUN_REPORT_TSV}"
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
        echo "# MT-Bench SMT/S2FT SACA-Style Instruction-Tuning Results"
        echo ""
        echo "- model: ${MODEL}"
        echo "- dataset: ${DATASET}"
        echo "- methods: ${METHODS}"
        echo "- learning_rates: ${LEARNING_RATES}"
        echo "- target_params: ${TARGET_PARAMS}"
        echo "- batch_size: ${BATCH_SIZE}"
        echo "- gradient_accumulation_steps: ${GRAD_ACCUM}"
        echo "- max_seq_length: ${MAX_SEQ_LENGTH}"
        echo "- source_max_len: ${SOURCE_MAX_LEN}"
        echo "- target_max_len: ${TARGET_MAX_LEN}"
        echo "- warmup_steps: ${WARMUP_STEPS}"
        echo "- warmup_ratio: ${WARMUP_RATIO}"
        echo "- train_conda_env: ${TRAIN_CONDA_ENV:-current}"
        echo "- finetune_script: ${FINETUNE_SCRIPT}"
        echo "- deepspeed_config: ${DS_CONFIG}"
        echo "- eval_script: ${LLM_JUDGE_SCRIPT}"
        echo "- judge_model: ${JUDGE_MODEL}"
        echo "- smt_calibration_steps: ${SMT_CALIBRATION_STEPS}"
        echo "- smt_calibration_batch_size: ${SMT_CALIBRATION_BATCH_SIZE}"
        echo "- smt_attention_target_modules: ${SMT_ATTENTION_TARGET_MODULES}"
        echo "- smt_mlp_target_modules: ${SMT_MLP_TARGET_MODULES}"
        echo "- smt_budget_allocation: ${SMT_BUDGET_ALLOCATION}"
        echo "- smt_selection_strategy: ${SMT_SELECTION_STRATEGY}"
        echo "- smt_calculation_strategy: ${SMT_CALCULATION_STRATEGY}"
        echo "- s2ft_calibration_steps: ${S2FT_CALIBRATION_STEPS}"
        echo "- s2ft_calibration_batch_size: ${S2FT_CALIBRATION_BATCH_SIZE}"
        echo "- s2ft_selection_method: ${S2FT_SELECTION_METHOD}"
        echo "- s2ft_ratio_preset: ${S2FT_RATIO_PRESET}"
        echo "- s2ft_layer_allocation: ${S2FT_LAYER_ALLOCATION}"
        echo "- s2ft_target_projections: ${S2FT_TARGET_PROJECTIONS}"
        echo "- bf16_quantize_flag: ${SACA_STYLE_QUANTIZE}"
        echo ""
    } > "${RESULTS}"
fi
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${METRICS_FILE}" ]; then
    echo -e "method\tlearning_rate\tphase\telapsed_seconds\tbatch_size\tgradient_accumulation_steps\tmax_seq_length\twarmup_steps\twarmup_ratio\toutput_dir\texit_code" > "${METRICS_FILE}"
fi
RUN_REPORT_HEADER="run_id\tmethod\tlearning_rate\trun_tag\ttarget_params\tbatch_size\tgradient_accumulation_steps\tmax_seq_length\tlr_scheduler\twarmup_steps\twarmup_ratio\tsmt_calibration_steps\tsmt_calibration_batch_size\tsmt_attention_target_modules\tsmt_mlp_target_modules\tsmt_budget_allocation\tsmt_selection_strategy\tsmt_calculation_strategy\ts2ft_calibration_steps\ts2ft_calibration_batch_size\ts2ft_selection_method\ts2ft_ratio_preset\ts2ft_layer_allocation\ts2ft_target_projections\tsource_max_len\ttarget_max_len\tbf16_quantize_flag\ttrain_elapsed_seconds\ttrain_exit\teval_elapsed_seconds\teval_exit\tturn1\tturn2\tavg\tcheckpoint"
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RUN_REPORT_TSV}" ]; then
    echo -e "${RUN_REPORT_HEADER}" > "${RUN_REPORT_TSV}"
fi

for METHOD in ${METHODS}; do
    for LR in ${LEARNING_RATES}; do
        LR_LABEL="$(lr_label "${LR}")"
        RUN_SUFFIX=""
        if [ -n "${RUN_TAG}" ]; then
            RUN_SUFFIX="_$(sanitize_tag "${RUN_TAG}")"
        fi
        RUN_ID="mtbench_${METHOD}_lr${LR_LABEL}${RUN_SUFFIX}"
        CKPT="${RESULT_ROOT}/checkpoints/${RUN_ID}"
        LOG="${RESULT_ROOT}/logs/${RUN_ID}"
        DATASET_CACHE="${RESULT_ROOT}/datasets/${RUN_ID}"
        mkdir -p "${CKPT}" "${LOG}" "${DATASET_CACHE}"

        TRAIN_ARGS=()
        if [ -n "${MAX_STEPS}" ]; then
            TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
        fi
        if [ -n "${HF_TOKEN}" ]; then
            TRAIN_ARGS+=(--hf_token "${HF_TOKEN}")
        fi
        [ "${SACA_STYLE_QUANTIZE}" = "true" ] && TRAIN_ARGS+=(--quantize)
        [ "${TRUST_REMOTE_CODE}" = "true" ] && TRAIN_ARGS+=(--trust_remote_code)
        [ "${TRAIN_ON_SOURCE}" = "true" ] && TRAIN_ARGS+=(--train_on_source)

        TRAIN_EXIT=0
        TRAIN_ELAPSED=""
        if [ "${RUN_TRAIN}" = "true" ]; then
            echo "========================================"
            echo "[MT-Bench] SACA-style training: method=${METHOD}, lr=${LR}"
            echo "========================================"
            PORT=$((29600 + RANDOM % 1000))
            TRAIN_START="$(date +%s)"
            set +e
            (
                cd "${SPARSE_FT_ROOT}"
                train_conda_run deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port="${PORT}" \
                    "${FINETUNE_SCRIPT}" \
                    --method "${METHOD}" \
                    --model "${MODEL}" \
                    --dataset "${DATASET}" \
                    --dataset_cache_dir "${DATASET_CACHE}" \
                    --save_dir "${CKPT}" \
                    --epochs "${EPOCHS}" \
                    --train_bs "${BATCH_SIZE}" \
                    --accumulation_steps "${GRAD_ACCUM}" \
                    --lr "${LR}" \
                    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
                    --warmup_steps "${WARMUP_STEPS}" \
                    --warmup_ratio "${WARMUP_RATIO}" \
                    --target_params "${TARGET_PARAMS}" \
                    --max_seq_length "${MAX_SEQ_LENGTH}" \
                    --source_max_len "${SOURCE_MAX_LEN}" \
                    --target_max_len "${TARGET_MAX_LEN}" \
                    --smt_calibration_steps "${SMT_CALIBRATION_STEPS}" \
                    --smt_calibration_batch_size "${SMT_CALIBRATION_BATCH_SIZE}" \
                    --s2ft_calibration_steps "${S2FT_CALIBRATION_STEPS}" \
                    --s2ft_calibration_batch_size "${S2FT_CALIBRATION_BATCH_SIZE}" \
                    --s2ft_selection_method "${S2FT_SELECTION_METHOD}" \
                    --seed 42 \
                    --run_project "oasst1_mtbench_smt_s2ft_saca_style" \
                    --run_name "${RUN_ID}" \
                    --run_id "${RUN_ID}" \
                    --report_to "${REPORT_TO}" \
                    --deepspeed "${DS_CONFIG}" \
                    --save_full_model \
                    "${TRAIN_ARGS[@]}"
            ) 2>&1 | tee "${LOG}/train.log"
            TRAIN_EXIT=${PIPESTATUS[0]}
            set -e
            TRAIN_ELAPSED=$(( $(date +%s) - TRAIN_START ))
            echo -e "${METHOD}\t${LR}\ttrain\t${TRAIN_ELAPSED}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${MAX_SEQ_LENGTH}\t${WARMUP_STEPS}\t${WARMUP_RATIO}\t${CKPT}\t${TRAIN_EXIT}" >> "${METRICS_FILE}"
        elif [ ! -f "${CKPT}/config.json" ]; then
            echo "[MT-Bench] ${METHOD} checkpoint not found for eval-only mode: ${CKPT}" | tee -a "${LOG}/train.log"
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
            echo -e "${METHOD}\t${LR}\tjudge\t${EVAL_ELAPSED}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${MAX_SEQ_LENGTH}\t${WARMUP_STEPS}\t${WARMUP_RATIO}\t${CKPT}\t${EVAL_EXIT}" >> "${METRICS_FILE}"
            if [ "${EVAL_EXIT}" -eq 0 ] && [ -f "${EVAL_SUMMARY_FILE}" ]; then
                SUMMARY_LINE="$(awk -F'\t' -v run_id="${RUN_ID}" '$2 == run_id { line=$0 } END { print line }' "${EVAL_SUMMARY_FILE}")"
                if [ -n "${SUMMARY_LINE}" ]; then
                    IFS=$'\t' read -r _ts _run_id _model_id _algorithm _lr _tag _judge TURN1 TURN2 AVG _model_path <<< "${SUMMARY_LINE}"
                fi
            fi
            echo "[MT-Bench] ${METHOD}, lr=${LR} llm_judge exit=${EVAL_EXIT}, avg=${AVG:-NA}"
        fi

        append_run_report "${RUN_ID}" "${METHOD}" "${LR}" "${CKPT}" "${TRAIN_ELAPSED}" "${TRAIN_EXIT}" "${EVAL_ELAPSED}" "${EVAL_EXIT}" "${TURN1}" "${TURN2}" "${AVG}"
    done
done

echo "========================================"
echo "[MT-Bench] ALL RUNS COMPLETE"
echo "Results: ${RESULTS}"
echo "Run report: ${RUN_REPORT_MD}"
echo "Judge summary: ${EVAL_SUMMARY_FILE}"
echo "========================================"
