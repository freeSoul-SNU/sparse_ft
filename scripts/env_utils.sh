#!/usr/bin/env bash

init_conda_shell() {
    local conda_root

    if [ -n "${CONDA_EXE:-}" ]; then
        conda_root="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
        if [ -f "${conda_root}/etc/profile.d/conda.sh" ]; then
            source "${conda_root}/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    for conda_root in /opt/miniconda3 /opt/anaconda3 "${HOME}/miniconda3" "${HOME}/anaconda3"; do
        if [ -f "${conda_root}/etc/profile.d/conda.sh" ]; then
            source "${conda_root}/etc/profile.d/conda.sh"
            return 0
        fi
    done

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return 0
    fi

    echo "conda command was not found. Install conda or set CONDA_EXE." >&2
    return 1
}

activate_sparse_ft_conda() {
    init_conda_shell

    local had_nounset=0
    case $- in
        *u*) had_nounset=1; set +u ;;
    esac

    if [ -n "${CONDA_ENV_PREFIX:-}" ] && [ -d "${CONDA_ENV_PREFIX}" ]; then
        conda activate "${CONDA_ENV_PREFIX}"
    else
        conda activate "${ENV_NAME:-sparse-ft}"
    fi

    if [ "${had_nounset}" -eq 1 ]; then
        set -u
    fi
}

configure_cuda_env() {
    local torch_cuda_major=""
    local candidate=""
    local nvcc_path=""

    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        export PATH="${CUDA_HOME}/bin:${PATH}"
        return 0
    fi

    torch_cuda_major="$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    cuda = torch.version.cuda or ""
    print(cuda.split(".")[0] if cuda else "")
except Exception:
    print("")
PY
)"

    if [ -n "${CUDA_HOME_CANDIDATES:-}" ]; then
        for candidate in ${CUDA_HOME_CANDIDATES}; do
            if [ -x "${candidate}/bin/nvcc" ]; then
                export CUDA_HOME="${candidate}"
                export PATH="${CUDA_HOME}/bin:${PATH}"
                echo "[cuda] CUDA_HOME=${CUDA_HOME}"
                return 0
            fi
        done
    fi

    if [ -n "${torch_cuda_major}" ]; then
        while IFS= read -r candidate; do
            if [ -x "${candidate}/bin/nvcc" ]; then
                export CUDA_HOME="${candidate}"
                export PATH="${CUDA_HOME}/bin:${PATH}"
                echo "[cuda] CUDA_HOME=${CUDA_HOME} (torch cuda major ${torch_cuda_major})"
                return 0
            fi
        done < <(find /usr/local -maxdepth 1 -type d -name "cuda-${torch_cuda_major}*" 2>/dev/null | sort -V)
    fi

    for candidate in /usr/local/cuda /usr/local/cuda-12.3 /usr/local/cuda-12.4 /usr/local/cuda-12.8 /usr/local/cuda-11.8; do
        if [ -x "${candidate}/bin/nvcc" ]; then
            export CUDA_HOME="${candidate}"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            echo "[cuda] CUDA_HOME=${CUDA_HOME}"
            return 0
        fi
    done

    nvcc_path="$(command -v nvcc 2>/dev/null || true)"
    if [ -n "${nvcc_path}" ]; then
        export CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        echo "[cuda] CUDA_HOME=${CUDA_HOME}"
        return 0
    fi

    echo "Could not find nvcc. Set CUDA_HOME to a CUDA installation with bin/nvcc." >&2
    return 1
}

_sf_parse_slurm_minutes() {
    local value="$1"
    local days=0
    local time_part
    local hours=0
    local minutes=0
    local seconds=0

    if [ -z "${value}" ] || [ "${value}" = "UNLIMITED" ] || [ "${value}" = "Partition_Limit" ]; then
        return 1
    fi

    if [[ "${value}" == *-* ]]; then
        days="${value%%-*}"
        time_part="${value#*-}"
    else
        time_part="${value}"
    fi

    IFS=':' read -r hours minutes seconds <<< "${time_part}"
    minutes="${minutes:-0}"
    seconds="${seconds:-0}"

    if ! [[ "${days}" =~ ^[0-9]+$ && "${hours}" =~ ^[0-9]+$ && "${minutes}" =~ ^[0-9]+$ && "${seconds}" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    echo $((days * 1440 + hours * 60 + minutes + (seconds > 0 ? 1 : 0)))
}

_sf_format_slurm_minutes() {
    local total_minutes="$1"
    local days
    local rem
    local hours
    local minutes

    if [ "${total_minutes}" -lt 1 ]; then
        total_minutes=1
    fi

    days=$((total_minutes / 1440))
    rem=$((total_minutes % 1440))
    hours=$((rem / 60))
    minutes=$((rem % 60))

    if [ "${days}" -gt 0 ]; then
        printf '%d-%02d:%02d:00\n' "${days}" "${hours}" "${minutes}"
    else
        printf '%02d:%02d:00\n' "${hours}" "${minutes}"
    fi
}

_sf_tres_minutes_value() {
    local tres="$1"
    local key="$2"
    local item

    IFS=',' read -ra _sf_tres_items <<< "${tres}"
    for item in "${_sf_tres_items[@]}"; do
        item="${item// /}"
        if [[ "${item}" == "${key}="* ]]; then
            echo "${item#*=}"
            return 0
        fi
    done

    return 1
}

_sf_query_partition_max_minutes() {
    local partition="$1"
    local raw

    if ! command -v sinfo >/dev/null 2>&1; then
        return 1
    fi

    raw="$(sinfo -h -p "${partition}" -o '%l' 2>/dev/null | awk 'NF { print; exit }')"
    _sf_parse_slurm_minutes "${raw}"
}

_sf_query_budget_gpu_minutes() {
    local account="$1"
    local user="${2:-${USER:-}}"
    local row
    local limit_tres
    local running_tres
    local limit
    local running=0

    if ! command -v sshare >/dev/null 2>&1; then
        return 1
    fi

    row="$(sshare -A "${account}" -u "${user}" -l -P 2>/dev/null \
        | awk -F'|' 'NR > 1 && $10 != "" { print $10 "|" $11; exit }')"
    if [ -z "${row}" ]; then
        return 1
    fi

    limit_tres="${row%%|*}"
    running_tres="${row#*|}"

    limit="$(_sf_tres_minutes_value "${limit_tres}" "gres/gpu" 2>/dev/null \
        || _sf_tres_minutes_value "${limit_tres}" "billing" 2>/dev/null \
        || true)"
    if [ -z "${limit}" ] || ! [[ "${limit}" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    running="$(_sf_tres_minutes_value "${running_tres}" "gres/gpu" 2>/dev/null \
        || _sf_tres_minutes_value "${running_tres}" "billing" 2>/dev/null \
        || echo 0)"
    if ! [[ "${running}" =~ ^[0-9]+$ ]]; then
        running=0
    fi

    echo $((limit - running))
}

_sf_resolve_slurm_time() {
    local requested_time="$1"
    local num_gpus="$2"
    local account="$3"
    local partition="$4"
    local fallback="${SLURM_TIME_FALLBACK:-20:00:00}"
    local safety_minutes="${SLURM_TIME_BUDGET_SAFETY_MINUTES:-10}"
    local partition_minutes=""
    local budget_gpu_minutes=""
    local budget_wall_minutes=""
    local chosen_minutes=""

    if [ "${requested_time}" != "auto" ]; then
        echo "${requested_time}"
        return 0
    fi

    partition_minutes="$(_sf_query_partition_max_minutes "${partition}" 2>/dev/null || true)"
    budget_gpu_minutes="$(_sf_query_budget_gpu_minutes "${account}" "${USER:-}" 2>/dev/null || true)"

    if [ -n "${budget_gpu_minutes}" ] && [[ "${budget_gpu_minutes}" =~ ^-?[0-9]+$ ]]; then
        budget_wall_minutes=$((budget_gpu_minutes / num_gpus - safety_minutes))
        if [ "${budget_wall_minutes}" -lt 1 ]; then
            echo "Not enough GPU budget for gpu:${num_gpus}. remaining_gpu_minutes=${budget_gpu_minutes}, safety_minutes=${safety_minutes}" >&2
            exit 1
        fi
        chosen_minutes="${budget_wall_minutes}"
    fi

    if [ -n "${partition_minutes}" ] && [[ "${partition_minutes}" =~ ^[0-9]+$ ]]; then
        if [ -z "${chosen_minutes}" ] || [ "${partition_minutes}" -lt "${chosen_minutes}" ]; then
            chosen_minutes="${partition_minutes}"
        fi
    fi

    if [ -z "${chosen_minutes}" ]; then
        echo "[srun] could not read partition/budget time; using fallback ${fallback}" >&2
        echo "${fallback}"
        return 0
    fi

    echo "[srun] auto time: partition_minutes=${partition_minutes:-unknown}, remaining_gpu_minutes=${budget_gpu_minutes:-unknown}, safety_minutes=${safety_minutes}, num_gpus=${num_gpus}" >&2
    _sf_format_slurm_minutes "${chosen_minutes}"
}

maybe_reexec_with_srun() {
    local script_path="$1"
    shift

    if [ "${RUN_WITH_SRUN:-false}" != "true" ] || [ "${SPARSE_FT_IN_SRUN:-0}" = "1" ]; then
        return 0
    fi
    if [ -n "${SLURM_STEP_ID:-}" ] && [ "${SLURM_STEP_ID}" != "extern" ]; then
        return 0
    fi

    local num_gpus="${NUM_GPUS:-1}"
    local account="${SLURM_ACCOUNT:-mms}"
    local partition="${SLURM_PARTITION:-gpu_default}"
    local cpus_per_gpu="${CPUS_PER_GPU:-6}"
    local requested_time="${SLURM_TIME:-auto}"
    local resolved_time

    if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]] || [ "${num_gpus}" -lt 1 ] || [ "${num_gpus}" -gt 2 ]; then
        echo "NUM_GPUS must be 1 or 2 on this server. Got: ${num_gpus}" >&2
        exit 1
    fi

    if ! command -v srun >/dev/null 2>&1; then
        echo "srun command was not found, but RUN_WITH_SRUN=true." >&2
        exit 1
    fi

    resolved_time="$(_sf_resolve_slurm_time "${requested_time}" "${num_gpus}" "${account}" "${partition}")"

    echo "[srun] requesting gpu:${num_gpus}, cpus-per-gpu:${cpus_per_gpu}, time:${resolved_time}"
    export SPARSE_FT_IN_SRUN=1
    exec srun \
        -A "${account}" \
        -p "${partition}" \
        --gres="gpu:${num_gpus}" \
        --cpus-per-gpu="${cpus_per_gpu}" \
        -t "${resolved_time}" \
        bash -lc 'cd "$1" && shift && exec "$@"' _ "${PWD}" "${script_path}" "$@"
}
