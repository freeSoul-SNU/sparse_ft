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
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-}"
# LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-1e-3 5e-4}}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-1e-3}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WARMUP_STEPS="${WARMUP_STEPS:-61}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RESET_RESULTS="${RESET_RESULTS:-false}"
SYNC_LMFLOW="${SYNC_LMFLOW:-auto}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

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
        echo "- smt_attention_target_modules: ${SMT_ATTENTION_TARGET_MODULES}"
        echo "- smt_budget_allocation: ${SMT_BUDGET_ALLOCATION}"
        echo "- s2ft_selection_method: ${S2FT_SELECTION_METHOD}"
        echo "- s2ft_target_projections: ${S2FT_TARGET_PROJECTIONS}"
        echo ""
    } > "${RESULTS}"
fi
if [ "${RESET_RESULTS}" = "true" ] || [ ! -f "${METRICS_FILE}" ]; then
    echo -e "method\tlearning_rate\tphase\telapsed_seconds\tbatch_size\tgradient_accumulation_steps\tmax_seq_length\twarmup_steps\toutput_dir\texit_code" > "${METRICS_FILE}"
fi

cd "${LMFLOW_DIR}"

for METHOD in ${METHODS}; do
    for LR in ${LEARNING_RATES}; do
        LR_LABEL="${LR//./p}"
        LR_LABEL="${LR_LABEL//+/_}"
        LR_LABEL="${LR_LABEL//-/_}"
        RUN_ID="mtbench_${METHOD}_lr${LR_LABEL}"
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

        if [ "${RUN_EVAL}" = "true" ]; then
            echo "[MT-Bench] Evaluating: method=${METHOD}, lr=${LR}"
            python "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/eval_mtbench.py" \
                --model_path "${CKPT}" \
                --method "${METHOD}" \
                --results_file "${RESULTS}" \
                --judge "${JUDGE_MODEL}" \
                --openai_api_key "${OPENAI_API_KEY}" \
                --num_gpus "${NUM_GPUS}" \
                2>&1 | tee "${LOG}/eval.log"
            echo "[MT-Bench] ${METHOD}, lr=${LR} evaluation DONE"
        fi
    done
done

echo "========================================"
echo "[MT-Bench] ALL RUNS COMPLETE"
echo "Results: ${RESULTS}"
echo "========================================"
