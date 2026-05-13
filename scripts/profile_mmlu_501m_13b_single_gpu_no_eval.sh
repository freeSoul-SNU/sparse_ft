#!/usr/bin/env bash

set -euo pipefail

# Single-GPU profiling wrapper for Llama-2-13B with ~501M trainable parameters.
# It reuses scripts/profile_mmlu_20m_dual_gpu_no_eval.sh, but sends selected
# methods to exactly one GPU and leaves the other GPU with no methods.

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"

export SPARSE_FT_ROOT
export LLM_ROOT
export RAPA_HOME

# Model / parameter budget.
export MODEL="${MODEL:-meta-llama/Llama-2-13b-hf}"
export TARGET_PARAMS="${TARGET_PARAMS:-501000000}"

# Keep results separate from the 7B/20M and dual-GPU 13B/501M profiles.
export RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/profile_mmlu_501m_13b_single_gpu_no_eval}"

# Safer defaults for 13B on a single A100-class GPU.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export EPOCHS="${EPOCHS:-1}"
export PROFILE_STEPS="${PROFILE_STEPS:-10}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"

# Single-GPU mode.  LT-SFT is excluded by default here because the current
# 13B/501M LT-SFT path is known to finish the 100-step dense mask search and
# then die with SIGKILL during post-search processing.  Re-include it explicitly
# after fixing that memory path.
export PROFILE_GPU="${PROFILE_GPU:-1}"
export PROFILE_METHODS="${PROFILE_METHODS:-sift smt spiel s2ft}"
if [ "${PROFILE_INCLUDE_LTSFT:-false}" = "true" ]; then
    export PROFILE_METHODS="${PROFILE_METHODS} ltsft"
fi

case "${PROFILE_GPU}" in
    0)
        export GPU0_METHODS="${GPU0_METHODS:-${PROFILE_METHODS}}"
        export GPU1_METHODS="${GPU1_METHODS-}"
        ;;
    1)
        export GPU0_METHODS="${GPU0_METHODS-}"
        export GPU1_METHODS="${GPU1_METHODS:-${PROFILE_METHODS}}"
        ;;
    *)
        echo "PROFILE_GPU must be 0 or 1, got: ${PROFILE_GPU}" >&2
        exit 1
        ;;
esac

# Use the paper-like SIFT hook path by default. It is single-GPU/no-DeepSpeed
# unless SIFT_HOOK_USE_DEEPSPEED=true is set; other 13B/501M methods keep CPU
# optimizer offload by default.
export SIFT_IMPLEMENTATION="${SIFT_IMPLEMENTATION:-hook}"
export SIFT_HOOK_USE_DEEPSPEED="${SIFT_HOOK_USE_DEEPSPEED:-false}"
export SPIEL_DELTA_DTYPE="${SPIEL_DELTA_DTYPE:-float32}"
export SPIEL_SELECTION_ALGORITHM="${SPIEL_SELECTION_ALGORITHM:-rigl}"
export SPIEL_RESELECTION_STEPS="${SPIEL_RESELECTION_STEPS:-20}"
export SPIEL_SELECTION_ACCUMULATION_STEPS="${SPIEL_SELECTION_ACCUMULATION_STEPS:-5}"
export SPIEL_RESELECTION_RATE_POLICY="${SPIEL_RESELECTION_RATE_POLICY:-linear}"
export SPIEL_INITIAL_RESELECTION_RATE="${SPIEL_INITIAL_RESELECTION_RATE:-0.2}"
export SPIEL_TARGET_MODULES="${SPIEL_TARGET_MODULES:-q_proj,o_proj,v_proj,k_proj,gate_proj,up_proj,down_proj}"
export SPIEL_STRIP_DS_OPTIMIZER="${SPIEL_STRIP_DS_OPTIMIZER:-true}"
export CPU_OFFLOAD_METHODS="${CPU_OFFLOAD_METHODS-smt ltsft spiel s2ft}"
export ENABLE_OFFLOAD_RETRY="${ENABLE_OFFLOAD_RETRY:-true}"

# Match the official SMT PEFT example by default: q/k/v attention blocks,
# no-restriction global top-k, mean_abs block scoring.
export SMT_TARGET_MODULES="${SMT_TARGET_MODULES:-q_proj,k_proj,v_proj}"
export SMT_SELECTION_STRATEGY="${SMT_SELECTION_STRATEGY:-no_restriction}"
export SMT_CALCULATION_STRATEGY="${SMT_CALCULATION_STRATEGY:-mean_abs}"
export SMT_BUDGET_ALLOCATION="${SMT_BUDGET_ALLOCATION:-attention_only}"

# Keep calibration microbatches small for memory profiling on 13B.
export SIFT_CALIBRATION_BATCH_SIZE="${SIFT_CALIBRATION_BATCH_SIZE:-1}"
export SMT_CALIBRATION_BATCH_SIZE="${SMT_CALIBRATION_BATCH_SIZE:-1}"
export S2FT_CALIBRATION_BATCH_SIZE="${S2FT_CALIBRATION_BATCH_SIZE:-1}"

# S2FT defaults for the 501M budget: use the official-code-style random
# structure selection over v/o/u/d, and let train_s2ft derive the ratio from
# TARGET_PARAMS.  Explicit ratios such as S2FT_D_RATIO=0.03 are for reproducing
# specific paper/example-script variants, not for the 501M budget.
export S2FT_SELECTION_METHOD="${S2FT_SELECTION_METHOD:-random}"
export S2FT_TARGET_PROJECTIONS="${S2FT_TARGET_PROJECTIONS:-v,o,u,d}"
export S2FT_V_RATIO="${S2FT_V_RATIO:-}"
export S2FT_O_RATIO="${S2FT_O_RATIO:-}"
export S2FT_U_RATIO="${S2FT_U_RATIO:-}"
export S2FT_D_RATIO="${S2FT_D_RATIO:-}"

# Call through bash so the base script does not need executable permission.
export RAPA_DATALOADER_NUM_WORKERS="${RAPA_DATALOADER_NUM_WORKERS:-0}"
export RAPA_DATALOADER_PIN_MEMORY="${RAPA_DATALOADER_PIN_MEMORY:-false}"
export RAPA_USE_DYNAMIC_PADDING="${RAPA_USE_DYNAMIC_PADDING:-true}"
exec bash "${SPARSE_FT_ROOT}/scripts/profile_mmlu_20m_dual_gpu_no_eval.sh"
