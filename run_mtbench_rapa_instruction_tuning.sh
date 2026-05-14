#!/bin/bash
# MT-Bench fine-tuning pipeline: RAPA instruction tuning or sparse FT over OASST1.
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
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_BASE}/LMFlow}"
INSTRUCTION_TUNING_DIR="${INSTRUCTION_TUNING_DIR:-${RAPA_BASE}/instruction-tuning}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
RAPA_TRAIN_CONDA_ENV="${RAPA_TRAIN_CONDA_ENV:-saca}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
DATASET="${DATASET:-${DATA:-oasst1}}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_BASE}/rapa_r602_mistral7b_mtbench_single_gpu}"
RESULTS="${RESULTS:-${RESULT_ROOT}/results.md}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"

METHODS="${METHODS:-rapa}"
TARGET_PARAMS="${TARGET_PARAMS:-170000000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
# BATCH_SIZE="${BATCH_SIZE:-8}"
# GRAD_ACCUM="${GRAD_ACCUM:-2}"
# LEARNING_RATE="${LEARNING_RATE:-}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-1e-3}}"
# LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-1e-4 5e-5 5e-4}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WARMUP_STEPS="${WARMUP_STEPS:-61}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RESET_RESULTS="${RESET_RESULTS:-false}"
SYNC_LMFLOW="${SYNC_LMFLOW:-auto}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini-2024-07-18}"
LLM_JUDGE_DIR="${LLM_JUDGE_DIR:-${LLM_ROOT}/FastChat_rebuttal/fastchat/llm_judge}"
LLM_JUDGE_SCRIPT="${LLM_JUDGE_SCRIPT:-${LLM_JUDGE_DIR}/0_sparse_ft.sh}"
LLM_JUDGE_GPU="${LLM_JUDGE_GPU:-${GPU_INDEX:-0}}"
LLM_JUDGE_CONDA_ENV="${LLM_JUDGE_CONDA_ENV:-llmjudge_iclr2026}"
EVAL_SUMMARY_FILE="${EVAL_SUMMARY_FILE:-${RESULT_ROOT}/mtbench_judge_summary.tsv}"
RUN_REPORT_TSV="${RUN_REPORT_TSV:-${RESULT_ROOT}/run_report.tsv}"
RUN_REPORT_MD="${RUN_REPORT_MD:-${RESULT_ROOT}/run_report.md}"
RUN_TAG="${RUN_TAG:-}"

# SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-100}"
# SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-20}"
SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-1}"
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

RAPA_RANK="${RAPA_RANK:-602}"
RAPA_ALPHA="${RAPA_ALPHA:-1}"
# The reference finetune.py treats any value other than lm_head/all as all non-lm_head linear layers.
# For Mistral/Llama this resolves to q,k,v,o,gate,up,down projections.
RAPA_TARGET_MODULES="${RAPA_TARGET_MODULES:-no_head}"
RAPA_WARMUP_RATIO="${RAPA_WARMUP_RATIO:-0.1}"
RAPA_RUN_PROJECT="${RAPA_RUN_PROJECT:-oasst1_mtbench_rapa}"
RAPA_MERGE_DEVICE="${RAPA_MERGE_DEVICE:-cpu}"
RAPA_OFFLINE="${RAPA_OFFLINE:-true}"
RAPA_WANDB_DISABLED="${RAPA_WANDB_DISABLED:-false}"
RAPA_WANDB_MODE="${RAPA_WANDB_MODE:-offline}"
RAPA_ADAPTER_ROOT="${RAPA_ADAPTER_ROOT:-${RESULT_ROOT}/adapters}"

RAPA_DATALOADER_NUM_WORKERS="${RAPA_DATALOADER_NUM_WORKERS:-0}"
RAPA_DATALOADER_PIN_MEMORY="${RAPA_DATALOADER_PIN_MEMORY:-true}"
RAPA_USE_DYNAMIC_PADDING="${RAPA_USE_DYNAMIC_PADDING:-true}"
RAPA_DATA_FORMAT="${RAPA_DATA_FORMAT:-instruction}"
RAPA_INSTRUCT_SOURCE_MAX_LEN="${RAPA_INSTRUCT_SOURCE_MAX_LEN:-768}"
RAPA_INSTRUCT_TARGET_MAX_LEN="${RAPA_INSTRUCT_TARGET_MAX_LEN:-256}"
RAPA_TRAIN_ON_SOURCE="${RAPA_TRAIN_ON_SOURCE:-false}"

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
export PEFT_DIR="${PEFT_DIR:-${RAPA_BASE}/peft}"
export PYTHONPATH="${PEFT_DIR}/src:${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
# export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/ds_config_zero0_no_offload.json}"
export SMT_CALIBRATION_STEPS SMT_CALIBRATION_BATCH_SIZE
export SMT_ATTENTION_TARGET_MODULES SMT_MLP_TARGET_MODULES
export SMT_BUDGET_ALLOCATION SMT_SELECTION_STRATEGY SMT_CALCULATION_STRATEGY
export S2FT_CALIBRATION_STEPS S2FT_CALIBRATION_BATCH_SIZE
export S2FT_SELECTION_METHOD S2FT_RATIO_PRESET S2FT_LAYER_ALLOCATION S2FT_TARGET_PROJECTIONS
export RAPA_RANK RAPA_ALPHA RAPA_TARGET_MODULES RAPA_WARMUP_RATIO RAPA_RUN_PROJECT
export RAPA_MERGE_DEVICE RAPA_OFFLINE RAPA_WANDB_DISABLED RAPA_WANDB_MODE RAPA_ADAPTER_ROOT
export RAPA_DATALOADER_NUM_WORKERS RAPA_DATALOADER_PIN_MEMORY RAPA_USE_DYNAMIC_PADDING
export RAPA_DATA_FORMAT RAPA_INSTRUCT_SOURCE_MAX_LEN RAPA_INSTRUCT_TARGET_MAX_LEN RAPA_TRAIN_ON_SOURCE
export RAPA_TRAIN_CONDA_ENV

rapa_conda_run() {
    if [ -n "${RAPA_TRAIN_CONDA_ENV}" ]; then
        WANDB_DISABLED="${RAPA_WANDB_DISABLED}" WANDB_MODE="${RAPA_WANDB_MODE}" \
            conda run --no-capture-output -n "${RAPA_TRAIN_CONDA_ENV}" "$@"
    else
        WANDB_DISABLED="${RAPA_WANDB_DISABLED}" WANDB_MODE="${RAPA_WANDB_MODE}" "$@"
    fi
}

sync_lmflow_sources() {
    local target_dir="${LMFLOW_DIR}/src/lmflow/pipeline/rapa"
    local target_config_dir="${LMFLOW_DIR}/configs/rapa"

    if [ "${SYNC_LMFLOW}" = "false" ]; then
        echo "[sync] skipping LMFlow sync (SYNC_LMFLOW=false)"
        return 0
    fi

    mkdir -p "${target_dir}" "${target_config_dir}"
    cp -r "${SPARSE_FT_ROOT}/pipeline/." "${target_dir}/"
    cp -r "${SPARSE_FT_ROOT}/configs/." "${target_config_dir}/"
}

resolve_dataset_path() {
    if [ "${DATASET}" = "oasst1" ]; then
        DATASET_PATH="${RAPA_BASE}/data/oasst1_lmflow.json"
        if [ ! -f "${DATASET_PATH}" ]; then
            echo "[mtbench] Preparing OASST1 LMFlow dataset at ${DATASET_PATH}"
            python "${SPARSE_FT_ROOT}/pipeline/prepare_datasets.py"
        fi
        return 0
    fi

    if [ -f "${DATASET}" ] || [ -d "${DATASET}" ]; then
        DATASET_PATH="${DATASET}"
        return 0
    fi

    echo "Dataset not found: ${DATASET}" >&2
    echo "Use DATASET=oasst1 or DATASET=/path/to/lmflow_text_only.json" >&2
    exit 1
}

mkdir -p "${RESULT_ROOT}/checkpoints" "${RESULT_ROOT}/logs"
sync_lmflow_sources
resolve_dataset_path

if [ ! -f "${MODEL}/config.json" ] && [[ "${MODEL}" = /* ]]; then
    echo "Model config not found: ${MODEL}/config.json" >&2
    exit 1
fi
if [ ! -f "${DS_CONFIG}" ]; then
    echo "DeepSpeed config not found: ${DS_CONFIG}" >&2
    exit 1
fi
if [[ " ${METHODS} " == *" rapa "* ]]; then
    if [ ! -f "${INSTRUCTION_TUNING_DIR}/finetune.py" ]; then
        echo "RAPA instruction tuning script not found: ${INSTRUCTION_TUNING_DIR}/finetune.py" >&2
        exit 1
    fi
    if [ ! -f "${INSTRUCTION_TUNING_DIR}/merge_lora.py" ]; then
        echo "RAPA merge script not found: ${INSTRUCTION_TUNING_DIR}/merge_lora.py" >&2
        exit 1
    fi
fi
if [ "${RUN_EVAL}" = "true" ] && [ ! -x "${LLM_JUDGE_SCRIPT}" ]; then
    echo "LLM judge script not found or not executable: ${LLM_JUDGE_SCRIPT}" >&2
    exit 1
fi

if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RESULTS}" ]; then
    {
        echo "# MT-Bench Mistral-7B-v0.3 RAPA Instruction-Tuning Results"
        echo ""
        echo "- model: ${MODEL}"
        echo "- dataset: ${DATASET_PATH}"
        echo "- instruction_tuning_dir: ${INSTRUCTION_TUNING_DIR}"
        echo "- target_params: ${TARGET_PARAMS}"
        echo "- methods: ${METHODS}"
        echo "- learning_rates: ${LEARNING_RATES}"
        echo "- rapa_train_conda_env: ${RAPA_TRAIN_CONDA_ENV:-current}"
        echo "- batch_size: ${BATCH_SIZE}"
        echo "- gradient_accumulation_steps: ${GRAD_ACCUM}"
        echo "- max_seq_length: ${MAX_SEQ_LENGTH}"
        echo "- warmup_steps: ${WARMUP_STEPS}"
        echo "- run_tag: ${RUN_TAG:-none}"
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
        echo "- rapa_rank: ${RAPA_RANK}"
        echo "- rapa_alpha: ${RAPA_ALPHA}"
        echo "- rapa_target_modules: ${RAPA_TARGET_MODULES}"
        echo "- rapa_warmup_ratio: ${RAPA_WARMUP_RATIO}"
        echo "- rapa_merge_device: ${RAPA_MERGE_DEVICE}"
        echo "- rapa_wandb_disabled: ${RAPA_WANDB_DISABLED}"
        echo "- rapa_wandb_mode: ${RAPA_WANDB_MODE}"
        echo "- data_format: ${RAPA_DATA_FORMAT}"
        echo "- train_on_source: ${RAPA_TRAIN_ON_SOURCE}"
        echo ""
    } > "${RESULTS}"
fi
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${METRICS_FILE}" ]; then
    echo -e "method\tlearning_rate\tphase\telapsed_seconds\tbatch_size\tgradient_accumulation_steps\tmax_seq_length\twarmup_steps\toutput_dir\texit_code" > "${METRICS_FILE}"
fi
RUN_REPORT_HEADER="run_id\tmethod\tlearning_rate\trun_tag\ttarget_params\tbatch_size\tgradient_accumulation_steps\tmax_seq_length\tlr_scheduler\twarmup_steps\tsmt_calibration_steps\tsmt_calibration_batch_size\tsmt_attention_target_modules\tsmt_mlp_target_modules\tsmt_budget_allocation\tsmt_selection_strategy\tsmt_calculation_strategy\ts2ft_calibration_steps\ts2ft_calibration_batch_size\ts2ft_selection_method\ts2ft_ratio_preset\ts2ft_layer_allocation\ts2ft_target_projections\tdata_format\ttrain_on_source\ttrain_elapsed_seconds\ttrain_exit\teval_elapsed_seconds\teval_exit\tturn1\tturn2\tavg\tcheckpoint\trapa_rank\trapa_alpha\trapa_target_modules\trapa_adapter"
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RUN_REPORT_TSV}" ]; then
    echo -e "${RUN_REPORT_HEADER}" > "${RUN_REPORT_TSV}"
fi

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
        echo "# MT-Bench Sparse-FT Run Report"
        echo ""
        echo "- result_root: ${RESULT_ROOT}"
        echo "- judge_summary: ${EVAL_SUMMARY_FILE}"
        echo "- generated_at: $(date --iso-8601=seconds)"
        echo ""
        echo "| run_id | method | lr | tag | settings | batch | grad_accum | seq | train_exit | eval_exit | turn1 | turn2 | avg | checkpoint |"
        echo "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        awk -F'\t' 'NR > 1 {
            settings = "smt_cal=" $11 ", smt_budget=" $15 ", smt_sel=" $16 ", smt_calc=" $17 ", s2ft_cal=" $18 ", s2ft_sel=" $20 ", s2ft_ratio=" $21 ", s2ft_layer=" $22 ", s2ft_target=" $23 ", rapa_rank=" ($34 == "" ? "-" : $34) ", rapa_alpha=" ($35 == "" ? "-" : $35) ", rapa_targets=" ($36 == "" ? "-" : $36)
            printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, ($4 == "" ? "-" : $4), settings, $6, $7, $8, $27, $29, ($30 == "" ? "-" : $30), ($31 == "" ? "-" : $31), ($32 == "" ? "-" : $32), $33)
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

    REPORT_TMP="${RUN_REPORT_TSV}.tmp"
    awk -F'\t' -v run_id="${run_id}" 'NR == 1 || $1 != run_id' "${RUN_REPORT_TSV}" > "${REPORT_TMP}"
    mv "${REPORT_TMP}" "${RUN_REPORT_TSV}"
    echo -e "${run_id}\t${method}\t${lr}\t${RUN_TAG}\t${TARGET_PARAMS}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${MAX_SEQ_LENGTH}\t${LR_SCHEDULER_TYPE}\t${WARMUP_STEPS}\t${SMT_CALIBRATION_STEPS}\t${SMT_CALIBRATION_BATCH_SIZE}\t${SMT_ATTENTION_TARGET_MODULES}\t${SMT_MLP_TARGET_MODULES}\t${SMT_BUDGET_ALLOCATION}\t${SMT_SELECTION_STRATEGY}\t${SMT_CALCULATION_STRATEGY}\t${S2FT_CALIBRATION_STEPS}\t${S2FT_CALIBRATION_BATCH_SIZE}\t${S2FT_SELECTION_METHOD}\t${S2FT_RATIO_PRESET}\t${S2FT_LAYER_ALLOCATION}\t${S2FT_TARGET_PROJECTIONS}\t${RAPA_DATA_FORMAT}\t${RAPA_TRAIN_ON_SOURCE}\t${train_elapsed}\t${train_exit}\t${eval_elapsed}\t${eval_exit}\t${turn1}\t${turn2}\t${avg}\t${checkpoint}\t${RAPA_RANK}\t${RAPA_ALPHA}\t${RAPA_TARGET_MODULES}\t${ADAPTER_DIR:-}" >> "${RUN_REPORT_TSV}"
    write_markdown_report
}

cd "${LMFLOW_DIR}"

for METHOD in ${METHODS}; do
    for LR in ${LEARNING_RATES}; do
        LR_LABEL="$(lr_label "${LR}")"
        RUN_SUFFIX=""
        if [ -n "${RUN_TAG}" ]; then
            RUN_SUFFIX="_$(sanitize_tag "${RUN_TAG}")"
        fi
        RUN_ID="mtbench_${METHOD}_lr${LR_LABEL}${RUN_SUFFIX}"
        CKPT="${RESULT_ROOT}/checkpoints/${RUN_ID}"
        ADAPTER_DIR="${RAPA_ADAPTER_ROOT}/${RUN_ID}"
        LOG="${RESULT_ROOT}/logs/${RUN_ID}"
        mkdir -p "${CKPT}" "${LOG}"

        EXTRA_ARGS=()
        if [ -n "${MAX_STEPS}" ]; then
            EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
        fi
        if [ -n "${HF_TOKEN}" ]; then
            EXTRA_ARGS+=(--hf_token "${HF_TOKEN}")
        fi

        TRAIN_EXIT=0
        TRAIN_ELAPSED=""
        if [ "${RUN_TRAIN}" = "true" ]; then
            echo "========================================"
            echo "[MT-Bench] Training: method=${METHOD}, lr=${LR}"
            echo "========================================"

            PORT=$((29600 + RANDOM % 1000))
            TRAIN_START="$(date +%s)"
            set +e
            if [ "${METHOD}" = "rapa" ]; then
                mkdir -p "${ADAPTER_DIR}"
                RAPA_EXTRA_ARGS=()
                if [ "${RAPA_OFFLINE}" = "true" ]; then
                    RAPA_EXTRA_ARGS+=(--offline)
                fi
                if [ -n "${MAX_STEPS}" ]; then
                    echo "[MT-Bench] MAX_STEPS=${MAX_STEPS} is ignored for RAPA instruction-tuning finetune.py"
                fi
                {
                    (
                        cd "${INSTRUCTION_TUNING_DIR}"
                        rapa_conda_run deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port="${PORT}" \
                            finetune.py \
                            --custom_mode rapa \
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
                            --save_dir "${ADAPTER_DIR}" \
                            --deepspeed "${DS_CONFIG}" \
                            --task instruct \
                            --dataset "${DATASET}" \
                            --epochs "${EPOCHS}" \
                            "${RAPA_EXTRA_ARGS[@]}"
                    )
                    FINETUNE_EXIT=$?
                    if [ ${FINETUNE_EXIT} -ne 0 ]; then
                        exit ${FINETUNE_EXIT}
                    fi
                    (
                        cd "${INSTRUCTION_TUNING_DIR}"
                        rapa_conda_run python merge_lora.py \
                            --model_name_or_path "${MODEL}" \
                            --lora_model_path "${ADAPTER_DIR}" \
                            --output_model_path "${CKPT}" \
                            --device "${RAPA_MERGE_DEVICE}"
                    )
                } 2>&1 | tee "${LOG}/train.log"
            else
                deepspeed --include="${DEEPSPEED_INCLUDE}" --master_port="${PORT}" \
                    "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py" \
                    --method "${METHOD}" \
                    --model_name_or_path "${MODEL}" \
                    --dataset_path "${DATASET_PATH}" \
                    --output_dir "${CKPT}" \
                    --num_train_epochs "${EPOCHS}" \
                    --per_device_train_batch_size "${BATCH_SIZE}" \
                    --gradient_accumulation_steps "${GRAD_ACCUM}" \
                    --learning_rate "${LR}" \
                    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
                    --warmup_steps "${WARMUP_STEPS}" \
                    --max_seq_length "${MAX_SEQ_LENGTH}" \
                    --target_params "${TARGET_PARAMS}" \
                    --smt_calibration_steps "${SMT_CALIBRATION_STEPS}" \
                    --smt_calibration_batch_size "${SMT_CALIBRATION_BATCH_SIZE}" \
                    --s2ft_calibration_steps "${S2FT_CALIBRATION_STEPS}" \
                    --s2ft_calibration_batch_size "${S2FT_CALIBRATION_BATCH_SIZE}" \
                    --s2ft_selection_method "${S2FT_SELECTION_METHOD}" \
                    --seed 42 \
                    "${EXTRA_ARGS[@]}" \
                    2>&1 | tee "${LOG}/train.log"
            fi
            TRAIN_EXIT=${PIPESTATUS[0]}
            set -e
            TRAIN_ELAPSED=$(( $(date +%s) - TRAIN_START ))
            echo -e "${METHOD}\t${LR}\ttrain\t${TRAIN_ELAPSED}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${MAX_SEQ_LENGTH}\t${WARMUP_STEPS}\t${CKPT}\t${TRAIN_EXIT}" >> "${METRICS_FILE}"
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
            echo -e "${METHOD}\t${LR}\tjudge\t${EVAL_ELAPSED}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${MAX_SEQ_LENGTH}\t${WARMUP_STEPS}\t${CKPT}\t${EVAL_EXIT}" >> "${METRICS_FILE}"
            if [ "${EVAL_EXIT}" -eq 0 ] && [ -f "${EVAL_SUMMARY_FILE}" ]; then
                SUMMARY_LINE="$(awk -F'\t' -v run_id="${RUN_ID}" '$2 == run_id { line=$0 } END { print line }' "${EVAL_SUMMARY_FILE}")"
                if [ -n "${SUMMARY_LINE}" ]; then
                    read -r TURN1 TURN2 AVG <<< "$(awk -F'\t' -v run_id="${RUN_ID}" '$2 == run_id { turn1=$8; turn2=$9; avg=$10 } END { print turn1, turn2, avg }' "${EVAL_SUMMARY_FILE}")"
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
