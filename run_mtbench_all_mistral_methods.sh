#!/usr/bin/env bash
# Orchestrate sparse-ft SMT/S2FT and RAPA-repo PaCA/RAPA Mistral MT-Bench runs.
set -euo pipefail

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SPARSE_FT_ROOT}/scripts/env_utils.sh"

LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_BASE="${RAPA_HOME:-${LLM_ROOT}/rapa}"
RAPA_IT_DIR="${RAPA_IT_DIR:-${RAPA_BASE}/instruction-tuning}"
RAPA_MISTRAL_SCRIPT="${RAPA_MISTRAL_SCRIPT:-${RAPA_IT_DIR}/scripts/run_mistral.sh}"
SPARSE_MT_SCRIPT="${SPARSE_MT_SCRIPT:-${SPARSE_FT_ROOT}/run_mtbench_smt_s2ft_saca_style.sh}"

GPU_INDEX="${GPU_INDEX:-0}"
LEARNING_RATES="${LEARNING_RATES:-1e-4 5e-5 5e-4}"
LEARNING_RATES_CSV="${LEARNING_RATES// /,}"

RUN_SPARSE_FT="${RUN_SPARSE_FT:-true}"
RUN_RAPA_TRAIN="${RUN_RAPA_TRAIN:-true}"
RUN_RAPA_EVAL="${RUN_RAPA_EVAL:-true}"

SPARSE_METHODS="${SPARSE_METHODS:-s2ft smt}"
RAPA_METHODS="${RAPA_METHODS:-paca rapa}"
RAPA_METHODS_CSV="${RAPA_METHODS// /,}"

RAPA_TRAIN_CONDA_ENV="${RAPA_TRAIN_CONDA_ENV:-rapa}"
LLM_JUDGE_CONDA_ENV="${LLM_JUDGE_CONDA_ENV:-llmjudge_iclr2026}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini-2024-07-18}"
LLM_JUDGE_DIR="${LLM_JUDGE_DIR:-${LLM_ROOT}/FastChat_rebuttal/fastchat/llm_judge}"
LLM_JUDGE_SCRIPT="${LLM_JUDGE_SCRIPT:-${LLM_JUDGE_DIR}/0_sparse_ft.sh}"

RAPA_RESULT_ROOT="${RAPA_RESULT_ROOT:-${RAPA_BASE}/paca_rapa_mistral7b_mtbench_single_gpu}"
RAPA_OUTPUT_BASE="${RAPA_OUTPUT_BASE:-output_models/oasst1/Mistral-7B-v0.3}"
RAPA_DATASET="${RAPA_DATASET:-oasst1}"
RAPA_MODEL="${RAPA_MODEL:-mistralai/Mistral-7B-v0.3}"
RAPA_SEEDS="${RAPA_SEEDS:-42}"
RAPA_PORT="${RAPA_PORT:-28600}"
RUN_TAG="${RUN_TAG:-}"

PACA_RANK="${PACA_RANK:-128}"
PACA_ALPHA="${PACA_ALPHA:-1}"
RAPA_RANK="${RAPA_RANK:-602}"
RAPA_ALPHA="${RAPA_ALPHA:-1}"

TRAIN_BS="${TRAIN_BS:-2}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
EPOCHS="${EPOCHS:-1}"
TARGET_MODULES="${TARGET_MODULES:-no_head}"

init_conda_shell

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

rank_for_algorithm() {
    case "$1" in
        paca) printf '%s\n' "${PACA_RANK}" ;;
        rapa) printf '%s\n' "${RAPA_RANK}" ;;
        *) printf '%s\n' "${DEFAULT_RANK:-30}" ;;
    esac
}

alpha_for_algorithm() {
    case "$1" in
        paca) printf '%s\n' "${PACA_ALPHA}" ;;
        rapa) printf '%s\n' "${RAPA_ALPHA}" ;;
        *) printf '%s\n' "${LORA_ALPHA:-1}" ;;
    esac
}

first_seed() {
    local seeds="${RAPA_SEEDS//,/ }"
    for seed in ${seeds}; do
        printf '%s\n' "${seed}"
        return 0
    done
    printf '42\n'
}

link_rapa_checkpoints_for_judge() {
    local seed
    local run_suffix=""
    seed="$(first_seed)"
    if [ -n "${RUN_TAG}" ]; then
        run_suffix="_$(sanitize_tag "${RUN_TAG}")"
    fi

    mkdir -p "${RAPA_RESULT_ROOT}/checkpoints" "${RAPA_RESULT_ROOT}/logs"

    for algorithm in ${RAPA_METHODS}; do
        local rank
        local alpha
        rank="$(rank_for_algorithm "${algorithm}")"
        alpha="$(alpha_for_algorithm "${algorithm}")"

        for lr in ${LEARNING_RATES}; do
            local label
            local exp_id
            local merged_model
            local judge_checkpoint

            label="$(lr_label "${lr}")"
            exp_id="mistral_7B_${algorithm}_${RAPA_DATASET}_seed_${seed}_${lr}_rank_${rank}_alpha_${alpha}"
            merged_model="${RAPA_IT_DIR}/${RAPA_OUTPUT_BASE}/merge_mistral_${exp_id}"
            judge_checkpoint="${RAPA_RESULT_ROOT}/checkpoints/mtbench_${algorithm}_lr${label}${run_suffix}"

            if [ ! -f "${merged_model}/config.json" ]; then
                echo "[all-mtbench] Missing merged model for judge: ${merged_model}" >&2
                exit 1
            fi
            ln -sfn "${merged_model}" "${judge_checkpoint}"
            echo "[all-mtbench] linked ${judge_checkpoint} -> ${merged_model}"
        done
    done
}

if [ ! -x "${SPARSE_MT_SCRIPT}" ]; then
    echo "Sparse MT-Bench script not found or not executable: ${SPARSE_MT_SCRIPT}" >&2
    exit 1
fi
if [ ! -x "${RAPA_MISTRAL_SCRIPT}" ]; then
    echo "RAPA Mistral script not found or not executable: ${RAPA_MISTRAL_SCRIPT}" >&2
    exit 1
fi
if [ "${RUN_RAPA_EVAL}" = "true" ] && [ ! -x "${LLM_JUDGE_SCRIPT}" ]; then
    echo "LLM judge script not found or not executable: ${LLM_JUDGE_SCRIPT}" >&2
    exit 1
fi

if [ "${RUN_SPARSE_FT}" = "true" ]; then
    echo "========================================"
    echo "[all-mtbench] Running sparse-ft SMT/S2FT"
    echo "[all-mtbench] methods=${SPARSE_METHODS}, learning_rates=${LEARNING_RATES}"
    echo "========================================"
    GPU_INDEX="${GPU_INDEX}" \
    METHODS="${SPARSE_METHODS}" \
    LEARNING_RATES="${LEARNING_RATES}" \
    bash "${SPARSE_MT_SCRIPT}"
fi

if [ "${RUN_RAPA_TRAIN}" = "true" ]; then
    echo "========================================"
    echo "[all-mtbench] Running RAPA repo PaCA/RAPA instruction tuning"
    echo "[all-mtbench] methods=${RAPA_METHODS}, learning_rates=${LEARNING_RATES}"
    echo "========================================"
    LEARNING_RATES="${LEARNING_RATES}" \
    ALGORITHMS="${RAPA_METHODS_CSV}" \
    DATASET="${RAPA_DATASET}" \
    MODEL="${RAPA_MODEL}" \
    PACA_RANK="${PACA_RANK}" \
    PACA_ALPHA="${PACA_ALPHA}" \
    RAPA_RANK="${RAPA_RANK}" \
    RAPA_ALPHA="${RAPA_ALPHA}" \
    TRAIN_BS="${TRAIN_BS}" \
    ACCUMULATION_STEPS="${ACCUMULATION_STEPS}" \
    EPOCHS="${EPOCHS}" \
    TARGET_MODULES="${TARGET_MODULES}" \
    OUTPUT_BASE="${RAPA_OUTPUT_BASE}" \
    SEEDS="${RAPA_SEEDS}" \
    conda run --no-capture-output -n "${RAPA_TRAIN_CONDA_ENV}" \
        bash "${RAPA_MISTRAL_SCRIPT}" "${GPU_INDEX}" "" "${RAPA_METHODS_CSV}" "${RAPA_PORT}"
fi

if [ "${RUN_RAPA_EVAL}" = "true" ]; then
    echo "========================================"
    echo "[all-mtbench] Running MT-Bench judge for RAPA repo outputs"
    echo "========================================"
    link_rapa_checkpoints_for_judge

    RESULT_ROOT="${RAPA_RESULT_ROOT}" \
    CONDA_ENV="${LLM_JUDGE_CONDA_ENV}" \
    JUDGE_MODEL="${JUDGE_MODEL}" \
    EVAL_SUMMARY_FILE="${RAPA_RESULT_ROOT}/mtbench_judge_summary.tsv" \
    RUN_TAG="${RUN_TAG}" \
    SKIP_MISSING=false \
    bash "${LLM_JUDGE_SCRIPT}" "${RAPA_METHODS_CSV}" "${GPU_INDEX}" "${LEARNING_RATES_CSV}" "${RAPA_RESULT_ROOT}" "${RUN_TAG}" \
        2>&1 | tee "${RAPA_RESULT_ROOT}/logs/mtbench_judge.log"
fi

echo "========================================"
echo "[all-mtbench] COMPLETE"
echo "Sparse script: ${SPARSE_MT_SCRIPT}"
echo "RAPA result root: ${RAPA_RESULT_ROOT}"
echo "========================================"
