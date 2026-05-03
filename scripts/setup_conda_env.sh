#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sparse-ft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RAPA_HOME="${RAPA_HOME:-/home1/irteam/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
PEFT_DIR="${PEFT_DIR:-${RAPA_HOME}/peft}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINEAR_SD_DIR="${REPO_ROOT}/methods/spiel/peft_sft/linear-sd"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command was not found."
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda env create -f "${REPO_ROOT}/environment.yml" -n "${ENV_NAME}"
fi

conda activate "${ENV_NAME}"
python --version >/dev/null
pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install \
    "transformers>=4.41,<4.50" \
    "accelerate>=0.30" \
    "datasets>=2.19" \
    "deepspeed>=0.14.4" \
    "trl==0.8.0" \
    "peft>=0.11" \
    "vllm>=0.4.3" \
    "lm-eval>=0.4.2" \
    "openai>=1.30" \
    "evaluate>=0.4.2" \
    "sentencepiece>=0.2.0" \
    "tqdm>=4.66" \
    "numpy>=1.26" \
    "scipy>=1.12" \
    "scikit-learn>=1.4"

mkdir -p "${RAPA_HOME}"

if [ ! -d "${LMFLOW_DIR}" ]; then
    git clone https://github.com/OptimalScale/LMFlow.git "${LMFLOW_DIR}"
elif [ ! -d "${LMFLOW_DIR}/.git" ]; then
    echo "Using existing non-git LMFlow directory: ${LMFLOW_DIR}"
fi

if [ ! -d "${PEFT_DIR}" ]; then
    git clone https://github.com/AlanAnsell/peft.git "${PEFT_DIR}"
elif [ ! -d "${PEFT_DIR}/.git" ]; then
    echo "Using existing non-git PEFT directory: ${PEFT_DIR}"
fi

pip install --no-deps -e "${LMFLOW_DIR}"

mkdir -p "${LMFLOW_DIR}/src/lmflow/pipeline/rapa" "${LMFLOW_DIR}/configs/rapa"
cp -r "${REPO_ROOT}/pipeline/." "${LMFLOW_DIR}/src/lmflow/pipeline/rapa/"
cp -r "${REPO_ROOT}/configs/." "${LMFLOW_DIR}/configs/rapa/"

if [ -d "${LINEAR_SD_DIR}" ]; then
    (cd "${LINEAR_SD_DIR}" && python setup.py install)
fi

cat <<EOF
Environment is ready.
- conda env: ${ENV_NAME}
- LMFlow: ${LMFLOW_DIR}
- PEFT fork: ${PEFT_DIR}

Before training, set:
  export HF_TOKEN=...
  export HF_HOME=${RAPA_HOME}/hf_cache
EOF
