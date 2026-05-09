#!/usr/bin/env bash
set -euo pipefail

# LT-SFT 13B/501M train-phase profile:
# - run the original LT-SFT dense mask-search/selection length by default
# - enter sparse fine-tuning
# - run MAX_STEPS sparse training steps
# - log sparse-train-only time and memory from pipeline/ltsft_tuner.py

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"

timestamp="$(date +%Y%m%d_%H%M%S)"
export RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profile_ltsft_13b_501m_train_phase_${timestamp}}"
mkdir -p "${RESULT_ROOT}"

export METHODS="${METHODS:-ltsft}"
export GPU_INDEX="${GPU_INDEX:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
export RUN_EVAL="${RUN_EVAL:-false}"
export MAX_STEPS="${MAX_STEPS:-100}"
export LTSFT_MASK_SEARCH_STEPS="${LTSFT_MASK_SEARCH_STEPS:-100}"
export LTSFT_N_FT_ITERATIONS="${LTSFT_N_FT_ITERATIONS:-1}"
export RAPA_SKIP_SAVE="${RAPA_SKIP_SAVE:-true}"
export RAPA_DATALOADER_NUM_WORKERS="${RAPA_DATALOADER_NUM_WORKERS:-0}"

# Keep the existing 13B wrapper defaults unless explicitly overridden:
# BATCH_SIZE=4, GRAD_ACCUM=2.  If this OOMs on a single A100, rerun with
# BATCH_SIZE=1 GRAD_ACCUM=8 for the same effective batch size.

echo "[profile-ltsft] RESULT_ROOT=${RESULT_ROOT}"
echo "[profile-ltsft] GPU_INDEX=${GPU_INDEX}"
echo "[profile-ltsft] BATCH_SIZE=${BATCH_SIZE:-4}"
echo "[profile-ltsft] GRAD_ACCUM=${GRAD_ACCUM:-2}"
echo "[profile-ltsft] LTSFT_MASK_SEARCH_STEPS=${LTSFT_MASK_SEARCH_STEPS}"
echo "[profile-ltsft] MAX_STEPS=${MAX_STEPS}"

exec bash "${SPARSE_FT_ROOT}/run_mmlu_501m_13b_single_gpu.sh"
