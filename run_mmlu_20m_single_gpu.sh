#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
SPARSE_FT_ROOT="${SPARSE_FT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RAPA_HOME="${RAPA_HOME:-/data/nksol0405/LLM/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DATASET="${DATASET:-${RAPA_HOME}/OwLore_Dataset/mmlu/mmlu.json}"
RESULT_ROOT="${RESULT_ROOT:-${RAPA_HOME}/sparse_ft_20m_mmlu_single_gpu}"
RESULTS_FILE="${RESULTS_FILE:-${RESULT_ROOT}/results.md}"
METHODS="${METHODS:-sift spiel smt s2ft ltsft}"
GPU_INDEX="${GPU_INDEX:-0}"
TARGET_PARAMS="${TARGET_PARAMS:-20000000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_EVAL="${RUN_EVAL:-true}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

if [ ! -f "${DATASET}" ]; then
    echo "Dataset not found: ${DATASET}"
    exit 1
fi

if [ -f /home/nksol0405/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/nksol0405/anaconda3/etc/profile.d/conda.sh
else
    export PATH="/home/nksol0405/anaconda3/condabin:${PATH}"
    eval "$(conda shell.bash hook)"
fi
conda activate "${ENV_NAME}"

if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    export CUDA_HOME="/usr/local/cuda-12.3"
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export HF_HOME="${HF_HOME:-${RAPA_HOME}/hf_cache}"
export RAPA_HOME
export SPARSE_FT_ROOT
export PEFT_DIR="${PEFT_DIR:-${RAPA_HOME}/peft}"
export PYTHONPATH="${LMFLOW_DIR}/src:${SPARSE_FT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export DS_CONFIG="${DS_CONFIG:-${LMFLOW_DIR}/configs/rapa/ds_zero1.json}"

mkdir -p "${RESULT_ROOT}/checkpoints" "${RESULT_ROOT}/logs"
cp -r "${SPARSE_FT_ROOT}/pipeline/." "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/"
cp -r "${SPARSE_FT_ROOT}/configs/." "${LMFLOW_DIR}/configs/rapa/"

echo "# MMLU 20M Sparse FT Single-GPU Results" > "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"
echo "- model: ${MODEL}" >> "${RESULTS_FILE}"
echo "- dataset: ${DATASET}" >> "${RESULTS_FILE}"
echo "- target_params: ${TARGET_PARAMS}" >> "${RESULTS_FILE}"
echo "- gpu: ${GPU_INDEX}" >> "${RESULTS_FILE}"
echo "- learning_rate: ${LEARNING_RATE}" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

cd "${LMFLOW_DIR}"

for method in ${METHODS}; do
    run_name="mmlu_20m_${method}"
    ckpt="${RESULT_ROOT}/checkpoints/${run_name}"
    log_dir="${RESULT_ROOT}/logs/${run_name}"
    mkdir -p "${ckpt}" "${log_dir}"

    extra_args=()
    if [ -n "${MAX_STEPS}" ]; then
        extra_args+=(--max_steps "${MAX_STEPS}")
    fi

    echo "============================================"
    echo "[mmlu] Training ${method} on GPU ${GPU_INDEX}"
    echo "============================================"
    port=$((29500 + RANDOM % 1000))

    set +e
    deepspeed --include=localhost:0 --master_port="${port}" \
        "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/train_method.py" \
        --method "${method}" \
        --model_name_or_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --output_dir "${ckpt}" \
        --num_train_epochs "${EPOCHS}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate "${LEARNING_RATE}" \
        --lr_scheduler_type cosine \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --target_params "${TARGET_PARAMS}" \
        --bf16 \
        --seed 42 \
        "${extra_args[@]}" \
        2>&1 | tee "${log_dir}/train.log"
    train_ec=${PIPESTATUS[0]}
    set -e

    if [ "${train_ec}" -ne 0 ]; then
        echo "| ${method} | train_failed:${train_ec} | | |" >> "${RESULTS_FILE}"
        echo "[mmlu] ${method} train failed with exit=${train_ec}"
        continue
    fi

    if [ "${RUN_EVAL}" = "true" ]; then
        echo "============================================"
        echo "[mmlu] Evaluating ${method}"
        echo "============================================"
        set +e
        python "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/eval_lmharness.py" \
            --model_path "${ckpt}" \
            --method "${method}" \
            --task mmlu \
            --num_fewshot 5 \
            --results_file "${RESULTS_FILE}" \
            --num_gpus 1 \
            2>&1 | tee "${log_dir}/eval.log"
        eval_ec=${PIPESTATUS[0]}
        set -e
        echo "[mmlu] ${method} eval exit=${eval_ec}"
    fi
done

echo "Done. Results: ${RESULTS_FILE}"
