#!/usr/bin/env bash

set -euo pipefail

# Dedicated profiling wrapper for Llama-2-13B with ~501M trainable parameters.
# It reuses scripts/profile_mmlu_20m_dual_gpu_no_eval.sh, which already handles
# DeepSpeed launch, GPU/CPU monitoring, metrics.tsv writing, and offload retry.

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"

export SPARSE_FT_ROOT
export LLM_ROOT
export RAPA_HOME

# Model / parameter budget.
export MODEL="${MODEL:-meta-llama/Llama-2-13b-hf}"
export TARGET_PARAMS="${TARGET_PARAMS:-501000000}"

# Put 13B/501M results in a separate directory so previous 7B/20M profiles are not mixed.
export RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profile_mmlu_501m_13b_dual_gpu_no_eval}"

# Safer defaults for 13B on A100-class GPUs. This preserves an effective batch size of 8
# while reducing per-microbatch activation memory from the original BATCH_SIZE=8 setting.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export EPOCHS="${EPOCHS:-1}"
export PROFILE_STEPS="${PROFILE_STEPS:-10}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"

# Run the same five sparse fine-tuning methods as the original dual-GPU profiler.
export GPU0_METHODS="${GPU0_METHODS:-sift smt ltsft}"
export GPU1_METHODS="${GPU1_METHODS:-spiel s2ft}"

# Use the paper-like SIFT hook path by default. It is single-GPU/no-DeepSpeed
# unless SIFT_HOOK_USE_DEEPSPEED=true is set; other 13B/501M methods keep CPU
# optimizer offload by default.
export SIFT_IMPLEMENTATION="${SIFT_IMPLEMENTATION:-hook}"
export SIFT_HOOK_USE_DEEPSPEED="${SIFT_HOOK_USE_DEEPSPEED:-false}"
export CPU_OFFLOAD_METHODS="${CPU_OFFLOAD_METHODS:-smt ltsft spiel s2ft}"
export ENABLE_OFFLOAD_RETRY="${ENABLE_OFFLOAD_RETRY:-true}"

# Match SMT's selection space to the other methods. smt_tuner.py otherwise defaults to
# attention projections only, while the other methods target attention + MLP projections.
# export SMT_TARGET_MODULES="${SMT_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Keep calibration microbatches small for memory profiling on 13B.
export SIFT_CALIBRATION_BATCH_SIZE="${SIFT_CALIBRATION_BATCH_SIZE:-1}"
export SMT_CALIBRATION_BATCH_SIZE="${SMT_CALIBRATION_BATCH_SIZE:-1}"
export S2FT_CALIBRATION_BATCH_SIZE="${S2FT_CALIBRATION_BATCH_SIZE:-1}"

exec "${SPARSE_FT_ROOT}/scripts/profile_mmlu_20m_dual_gpu_no_eval.sh"
