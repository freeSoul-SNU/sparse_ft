#!/usr/bin/env bash
set -euo pipefail

# Wrapper for actual MMLU sparse fine-tuning:
# - Llama-2-13B
# - target trainable parameters: ~501M
# - single GPU
# - batch size 4, gradient accumulation 2
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_mmlu_501m_13b_single_gpu.sh
#
# Optional examples:
#   METHODS="sift smt" RUN_EVAL=false CUDA_VISIBLE_DEVICES=1 bash run_mmlu_501m_13b_single_gpu.sh
#   HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 bash run_mmlu_501m_13b_single_gpu.sh

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"

export NUM_GPUS="${NUM_GPUS:-1}"

# Model / dataset / output
export MODEL="${MODEL:-meta-llama/Llama-2-13b-hf}"
export DATASET="${DATASET:-${RAPA_HOME}/OwLore_Dataset/mmlu/mmlu.json}"
export RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/sparse_ft_501m_13b_mmlu_single_gpu}"
export RESULTS_FILE="${RESULTS_FILE:-${RESULT_ROOT}/results.md}"
export METRICS_FILE="${METRICS_FILE:-${RESULT_ROOT}/metrics.tsv}"

# Methods to run. Override to test a subset, e.g. METHODS="sift smt".
# export METHODS="${METHODS:-sift spiel smt s2ft ltsft}"
export METHODS="${METHODS:-spiel}"

# GPU selection:
# Prefer CUDA_VISIBLE_DEVICES from the command line.
# If it is not set, this uses physical GPU_INDEX.
# export GPU_INDEX="${GPU_INDEX:-0}"
export GPU_INDEX="${GPU_INDEX:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"

# Fine-tuning scale
export TARGET_PARAMS="${TARGET_PARAMS:-501000000}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-}"

# Requested batch setting
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# 13B + 501M trainable parameters can exceed a single A100's memory once
# optimizer states are included.  Use the same CPU optimizer offload path as
# the profiling scripts by default; override to CPU_OFFLOAD_METHODS='' only
# when intentionally testing the no-offload path.
export CPU_OFFLOAD_METHODS="${CPU_OFFLOAD_METHODS:-sift smt ltsft spiel s2ft}"

# Actual fine-tuning script defaults to train+eval.
# Set RUN_EVAL=false if you only want fine-tuning.
export RUN_TRAIN="${RUN_TRAIN:-true}"
export RUN_EVAL="${RUN_EVAL:-true}"

# Calibration / mask-search defaults inherited from the 20M script.
# Override if needed.
export SIFT_USE_GRADIENT_CALIBRATION="${SIFT_USE_GRADIENT_CALIBRATION:-true}"
export SIFT_CALIBRATION_ONLY="${SIFT_CALIBRATION_ONLY:-false}"
export SIFT_CALIBRATION_STEPS="${SIFT_CALIBRATION_STEPS:-1}"
export SIFT_CALIBRATION_BATCH_SIZE="${SIFT_CALIBRATION_BATCH_SIZE:-1}"
export SMT_CALIBRATION_STEPS="${SMT_CALIBRATION_STEPS:-100}"
export SMT_CALIBRATION_BATCH_SIZE="${SMT_CALIBRATION_BATCH_SIZE:-1}"
export S2FT_CALIBRATION_STEPS="${S2FT_CALIBRATION_STEPS:-100}"
export S2FT_CALIBRATION_BATCH_SIZE="${S2FT_CALIBRATION_BATCH_SIZE:-1}"
export LTSFT_MASK_SEARCH_STEPS="${LTSFT_MASK_SEARCH_STEPS:-100}"
export LTSFT_N_FT_ITERATIONS="${LTSFT_N_FT_ITERATIONS:-1}"

# For fairer SMT comparison: include attention + MLP projections.
# The SMT implementation reads this environment variable.
export SMT_TARGET_MODULES="${SMT_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Useful defaults
export RESET_RESULTS="${RESET_RESULTS:-false}"
export RESUME_TRAINING="${RESUME_TRAINING:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

echo "[run] MODEL=${MODEL}"
echo "[run] TARGET_PARAMS=${TARGET_PARAMS}"
echo "[run] METHODS=${METHODS}"
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] BATCH_SIZE=${BATCH_SIZE}"
echo "[run] GRAD_ACCUM=${GRAD_ACCUM}"
echo "[run] CPU_OFFLOAD_METHODS=${CPU_OFFLOAD_METHODS}"
echo "[run] RESULT_ROOT=${RESULT_ROOT}"

# Call the existing actual fine-tuning script through bash so it does not need executable permission.
exec bash "${SPARSE_FT_ROOT}/run_mmlu_20m_single_gpu.sh"
