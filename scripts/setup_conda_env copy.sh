#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rapa_h200}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env_utils.sh"

LLM_ROOT="${LLM_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
RAPA_HOME="${RAPA_HOME:-${LLM_ROOT}/rapa}"
LMFLOW_DIR="${LMFLOW_DIR:-${RAPA_HOME}/LMFlow}"
PEFT_DIR="${PEFT_DIR:-${RAPA_HOME}/peft}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${RAPA_HOME}/conda_envs/${ENV_NAME}}"
LINEAR_SD_DIR="${REPO_ROOT}/methods/spiel/peft_sft/linear-sd"

mkdir -p "${RAPA_HOME}" "$(dirname "${CONDA_ENV_PREFIX}")"
init_conda_shell

if [ ! -d "${CONDA_ENV_PREFIX}" ]; then
    conda env create -f "${REPO_ROOT}/environment.yml" -p "${CONDA_ENV_PREFIX}"
fi

conda activate "${CONDA_ENV_PREFIX}"
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
    (cd "${LINEAR_SD_DIR}" && CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}" python setup.py build_ext --inplace)
fi

cat <<EOF
Environment is ready.
- conda env: ${ENV_NAME}
- conda prefix: ${CONDA_ENV_PREFIX}
- LMFlow: ${LMFLOW_DIR}
- PEFT fork: ${PEFT_DIR}

Before training, set:
  export HF_TOKEN=...
  export HF_HOME=${RAPA_HOME}/hf_cache
EOF
