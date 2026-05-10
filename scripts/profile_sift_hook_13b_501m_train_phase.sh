#!/usr/bin/env bash
set -euo pipefail

# SIFT 13B/501M train-phase profile using the paper-like hook path:
# gradient top-k calibration, sparse optimizer parameters, and in-place merge
# back into the original Linear weights after optimizer steps.  It runs on GPU 1
# by default and stops after MAX_STEPS sparse fine-tuning steps.

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"

timestamp="$(date +%Y%m%d_%H%M%S)"
export RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profile_sift_hook_13b_501m_train_phase_${timestamp}}"
mkdir -p "${RESULT_ROOT}"

export METHODS="${METHODS:-sift}"
export GPU_INDEX="${GPU_INDEX:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
export RUN_EVAL="${RUN_EVAL:-false}"
export MAX_STEPS="${MAX_STEPS:-100}"
export TARGET_PARAMS="${TARGET_PARAMS:-501000000}"

export SIFT_IMPLEMENTATION="${SIFT_IMPLEMENTATION:-hook}"
export SIFT_USE_GRADIENT_CALIBRATION="${SIFT_USE_GRADIENT_CALIBRATION:-true}"
export SIFT_CALIBRATION_STEPS="${SIFT_CALIBRATION_STEPS:-1}"
export SIFT_CALIBRATION_BATCH_SIZE="${SIFT_CALIBRATION_BATCH_SIZE:-1}"
export SIFT_HOOK_ZERO_DENSE_GRAD="${SIFT_HOOK_ZERO_DENSE_GRAD:-true}"
export SIFT_HOOK_USE_DEEPSPEED="${SIFT_HOOK_USE_DEEPSPEED:-false}"

export RAPA_SKIP_SAVE="${RAPA_SKIP_SAVE:-true}"
export RAPA_DATALOADER_NUM_WORKERS="${RAPA_DATALOADER_NUM_WORKERS:-0}"

echo "[profile-sift] RESULT_ROOT=${RESULT_ROOT}"
echo "[profile-sift] GPU_INDEX=${GPU_INDEX}"
echo "[profile-sift] BATCH_SIZE=${BATCH_SIZE:-4}"
echo "[profile-sift] GRAD_ACCUM=${GRAD_ACCUM:-2}"
echo "[profile-sift] TARGET_PARAMS=${TARGET_PARAMS}"
echo "[profile-sift] SIFT_IMPLEMENTATION=${SIFT_IMPLEMENTATION}"
echo "[profile-sift] SIFT_CALIBRATION_STEPS=${SIFT_CALIBRATION_STEPS}"
echo "[profile-sift] SIFT_HOOK_USE_DEEPSPEED=${SIFT_HOOK_USE_DEEPSPEED}"
echo "[profile-sift] MAX_STEPS=${MAX_STEPS}"

exec bash "${SPARSE_FT_ROOT}/run_mmlu_501m_13b_single_gpu.sh"
