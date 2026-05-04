#!/usr/bin/env bash
set -euo pipefail

SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLM_ROOT="${LLM_ROOT:-$(cd "${SPARSE_FT_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/sparse_ft_20m_mmlu_single_gpu}"
SOURCE_SESSION="${SOURCE_SESSION:-sparse_ft_mmlu_20m}"
NEXT_SESSION="${NEXT_SESSION:-sparse_ft_mmlu_20m_batch8}"
RUN_LOG="${RUN_LOG:-${RESULT_ROOT}/run.log}"
NEXT_LOG="${NEXT_LOG:-${RESULT_ROOT}/run_batch8_remaining.log}"
SENTINEL="${SENTINEL:-[mmlu] spiel eval exit=}"

mkdir -p "${RESULT_ROOT}"
echo "[switch] waiting for SpiEL eval sentinel in ${RUN_LOG}: ${SENTINEL}"

while true; do
    if [ -f "${RUN_LOG}" ] && grep -Fq "${SENTINEL}" "${RUN_LOG}"; then
        break
    fi
    if ! tmux has-session -t "${SOURCE_SESSION}" 2>/dev/null; then
        echo "[switch] source session ${SOURCE_SESSION} no longer exists; continuing with batch8 resume"
        break
    fi
    sleep 10
done

echo "[switch] SpiEL boundary reached at $(date --iso-8601=seconds)"
if tmux has-session -t "${SOURCE_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SOURCE_SESSION}"
    echo "[switch] killed source session ${SOURCE_SESSION}"
fi

if tmux has-session -t "${NEXT_SESSION}" 2>/dev/null; then
    echo "[switch] next session ${NEXT_SESSION} already exists; not launching duplicate"
    exit 0
fi

tmux new-session -d -s "${NEXT_SESSION}" \
    "cd '${SPARSE_FT_ROOT}' && env RESULT_ROOT='${RESULT_ROOT}' METHODS='smt s2ft ltsft' BATCH_SIZE=8 GRAD_ACCUM=1 LEARNING_RATE=1e-4 RESET_RESULTS=false RUN_EVAL=true GPU_INDEX=0 LM_EVAL_TIMEOUT=21600 bash run_mmlu_20m_single_gpu_batch8_resume.sh > '${NEXT_LOG}' 2>&1"
echo "[switch] launched ${NEXT_SESSION}"
echo "[switch] log: ${NEXT_LOG}"
