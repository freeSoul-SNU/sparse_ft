#!/usr/bin/env bash
set -euo pipefail

# S2FT 13B/501M train-phase profile using S2FT-R defaults:
# random structured head/channel selection, uniform layer allocation, and
# projection ratios adjusted to match TARGET_PARAMS.
# This runs LMFlow/RAPA training on GPU 1 by default and stops after MAX_STEPS
# sparse fine-tuning steps while keeping the requested batch configuration.

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"

timestamp="$(date +%Y%m%d_%H%M%S)"
export RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profile_s2ft_13b_501m_train_phase_${timestamp}}"
mkdir -p "${RESULT_ROOT}"

export METHODS="${METHODS:-s2ft}"
export GPU_INDEX="${GPU_INDEX:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
export RUN_EVAL="${RUN_EVAL:-false}"
export MAX_STEPS="${MAX_STEPS:-100}"
export TARGET_PARAMS="${TARGET_PARAMS:-501000000}"
export S2FT_RATIO_PRESET="${S2FT_RATIO_PRESET:-budget}"
export S2FT_LAYER_ALLOCATION="${S2FT_LAYER_ALLOCATION:-uniform}"
export S2FT_SELECTION_METHOD="${S2FT_SELECTION_METHOD:-random}"
export RAPA_SKIP_SAVE="${RAPA_SKIP_SAVE:-true}"
export RAPA_DATALOADER_NUM_WORKERS="${RAPA_DATALOADER_NUM_WORKERS:-0}"

echo "[profile-s2ft] RESULT_ROOT=${RESULT_ROOT}"
echo "[profile-s2ft] GPU_INDEX=${GPU_INDEX}"
echo "[profile-s2ft] BATCH_SIZE=${BATCH_SIZE:-4}"
echo "[profile-s2ft] GRAD_ACCUM=${GRAD_ACCUM:-2}"
echo "[profile-s2ft] TARGET_PARAMS=${TARGET_PARAMS}"
echo "[profile-s2ft] S2FT_RATIO_PRESET=${S2FT_RATIO_PRESET}"
echo "[profile-s2ft] S2FT_LAYER_ALLOCATION=${S2FT_LAYER_ALLOCATION}"
echo "[profile-s2ft] S2FT_SELECTION_METHOD=${S2FT_SELECTION_METHOD}"
echo "[profile-s2ft] MAX_STEPS=${MAX_STEPS}"

exec bash "${SPARSE_FT_ROOT}/run_mmlu_501m_13b_single_gpu.sh"
