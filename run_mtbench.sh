#!/bin/bash
# MT-Bench sparse fine-tuning pipeline: SMT/S2FT over OASST1.
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
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
DATASET="${DATASET:-${DATA:-oasst1}}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_BASE}/sparse_ft_170m_mistral7b_mtbench_single_gpu}"
RESULTS="${RESULTS:-${RESULT_ROOT}/results.md}"
METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"

METHODS="${METHODS:-smt s2ft}"
TARGET_PARAMS="${TARGET_PARAMS:-170000000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-5e-5}}"
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

SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-100}"
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
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"
export SMT_CALIBRATION_STEPS SMT_CALIBRATION_BATCH_SIZE
export SMT_ATTENTION_TARGET_MODULES SMT_MLP_TARGET_MODULES
export SMT_BUDGET_ALLOCATION SMT_SELECTION_STRATEGY SMT_CALCULATION_STRATEGY
export S2FT_CALIBRATION_STEPS S2FT_CALIBRATION_BATCH_SIZE
export S2FT_SELECTION_METHOD S2FT_RATIO_PRESET S2FT_LAYER_ALLOCATION S2FT_TARGET_PROJECTIONS
export RAPA_DATALOADER_NUM_WORKERS RAPA_DATALOADER_PIN_MEMORY RAPA_USE_DYNAMIC_PADDING
export RAPA_DATA_FORMAT RAPA_INSTRUCT_SOURCE_MAX_LEN RAPA_INSTRUCT_TARGET_MAX_LEN RAPA_TRAIN_ON_SOURCE

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
if [ "${RUN_EVAL}" = "true" ] && [ ! -x "${LLM_JUDGE_SCRIPT}" ]; then
    echo "LLM judge script not found or not executable: ${LLM_JUDGE_SCRIPT}" >&2
    exit 1
fi

if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RESULTS}" ]; then
    {
        echo "# MT-Bench 170M Mistral-7B-v0.3 Sparse FT Single-GPU Results"
        echo ""
        echo "- model: ${MODEL}"
        echo "- dataset: ${DATASET_PATH}"
        echo "- target_params: ${TARGET_PARAMS}"
        echo "- methods: ${METHODS}"
        echo "- learning_rates: ${LEARNING_RATES}"
        echo "- batch_size: ${BATCH_SIZE}"
        echo "- gradient_accumulation_steps: ${GRAD_ACCUM}"
        echo "- max_seq_length: ${MAX_SEQ_LENGTH}"
        echo "- warmup_steps: ${WARMUP_STEPS}"
        echo "- run_tag: ${RUN_TAG:-none}"
        echo "- eval_script: ${LLM_JUDGE_SCRIPT}"
        echo "- judge_model: ${JUDGE_MODEL}"
        echo "- smt_attention_target_modules: ${SMT_ATTENTION_TARGET_MODULES}"
        echo "- smt_mlp_target_modules: ${SMT_MLP_TARGET_MODULES}"
        echo "- smt_budget_allocation: ${SMT_BUDGET_ALLOCATION}"
        echo "- s2ft_selection_method: ${S2FT_SELECTION_METHOD}"
        echo "- s2ft_target_projections: ${S2FT_TARGET_PROJECTIONS}"
        echo "- data_format: ${RAPA_DATA_FORMAT}"
        echo "- train_on_source: ${RAPA_TRAIN_ON_SOURCE}"
        echo ""
    } > "${RESULTS}"
fi
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${METRICS_FILE}" ]; then
    echo -e "method\tlearning_rate\tphase\telapsed_seconds\tbatch_size\tgradient_accumulation_steps\tmax_seq_length\twarmup_steps\toutput_dir\texit_code" > "${METRICS_FILE}"
fi
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${RUN_REPORT_TSV}" ]; then
    echo -e "run_id\tmethod\tlearning_rate\trun_tag\ttarget_params\tbatch_size\tgradient_accumulation_steps\tmax_seq_length\tlr_scheduler\twarmup_steps\tsmt_attention_target_modules\tsmt_mlp_target_modules\tsmt_budget_allocation\ts2ft_selection_method\ts2ft_target_projections\tdata_format\ttrain_on_source\ttrain_elapsed_seconds\ttrain_exit\teval_elapsed_seconds\teval_exit\tturn1\tturn2\tavg\tcheckpoint" > "${RUN_REPORT_TSV}"
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
        echo "| run_id | method | lr | tag | batch | grad_accum | seq | train_exit | eval_exit | turn1 | turn2 | avg | checkpoint |"
        echo "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        awk -F'\t' 'NR > 1 {
            printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, ($4 == "" ? "-" : $4), $6, $7, $8, $19, $21, ($22 == "" ? "-" : $22), ($23 == "" ? "-" : $23), ($24 == "" ? "-" : $24), $25)
        }' "${RUN_REPORT_TSV}"
    } > "${RUN_REPORT_MD}"
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
                --bf16 \
                --seed 42 \
                "${EXTRA_ARGS[@]}" \
                2>&1 | tee "${LOG}/train.log"
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
                    IFS=$'\t' read -r _ts _run_id _model_id _algorithm _lr _tag _judge TURN1 TURN2 AVG _model_path <<< "${SUMMARY_LINE}"
                fi
            fi
            echo "[MT-Bench] ${METHOD}, lr=${LR} llm_judge exit=${EVAL_EXIT}, avg=${AVG:-NA}"
        fi

        REPORT_TMP="${RUN_REPORT_TSV}.tmp"
        awk -F'\t' -v run_id="${RUN_ID}" 'NR == 1 || $1 != run_id' "${RUN_REPORT_TSV}" > "${REPORT_TMP}"
        mv "${REPORT_TMP}" "${RUN_REPORT_TSV}"
        echo -e "${RUN_ID}\t${METHOD}\t${LR}\t${RUN_TAG}\t${TARGET_PARAMS}\t${BATCH_SIZE}\t${GRAD_ACCUM}\t${MAX_SEQ_LENGTH}\t${LR_SCHEDULER_TYPE}\t${WARMUP_STEPS}\t${SMT_ATTENTION_TARGET_MODULES}\t${SMT_MLP_TARGET_MODULES}\t${SMT_BUDGET_ALLOCATION}\t${S2FT_SELECTION_METHOD}\t${S2FT_TARGET_PROJECTIONS}\t${RAPA_DATA_FORMAT}\t${RAPA_TRAIN_ON_SOURCE}\t${TRAIN_ELAPSED}\t${TRAIN_EXIT}\t${EVAL_ELAPSED}\t${EVAL_EXIT}\t${TURN1}\t${TURN2}\t${AVG}\t${CKPT}" >> "${RUN_REPORT_TSV}"
        write_markdown_report
    done
done

echo "========================================"
echo "[MT-Bench] ALL RUNS COMPLETE"
echo "Results: ${RESULTS}"
echo "Run report: ${RUN_REPORT_MD}"
echo "Judge summary: ${EVAL_SUMMARY_FILE}"
echo "========================================"
