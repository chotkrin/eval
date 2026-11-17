#!/usr/bin/env bash
# Shared SYNKA/Slurm implementation for LLM-jp, Phi and DSDT.
# Do not edit experiment lists here; edit EXP_NAMES in the model-specific submitter.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# sbatch copies this script into a Slurm spool directory before executing it.
# BASH_SOURCE[0] on workers therefore points into /var/spool, NOT the checkout.
# The submitter supplies the real root via --export; never recompute it there.
if [[ "${1:-}" == "__worker" ]]; then
    [[ -n "${PROJECT_ROOT:-}" && "${PROJECT_ROOT}" == /* ]] || {
        echo "ERROR: worker requires an absolute PROJECT_ROOT from the submitter" >&2
        exit 2
    }
    [[ -d "${PROJECT_ROOT}" ]] || {
        echo "ERROR: submitted PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
        exit 2
    }
    PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd -P)"
else
    PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
fi
[[ -f "${PROJECT_ROOT}/data_loader.py" ]] || {
    echo "ERROR: PROJECT_ROOT is not the evaluation checkout: ${PROJECT_ROOT}" >&2
    exit 2
}
# Load machine-local configuration on both submit and worker nodes.
ENV_FILE="${EVAL_ENV_FILE:-${PROJECT_ROOT}/.env_synka}"
[[ "${ENV_FILE}" == /* ]] || { echo "ERROR: EVAL_ENV_FILE must be absolute" >&2; exit 2; }
[[ -r "${ENV_FILE}" ]] || { echo "ERROR: cannot read SYNKA configuration ${ENV_FILE}" >&2; exit 2; }
if [[ -r "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

VERL_HOME="${VERL_HOME:?Set VERL_HOME in .env_synka}"
[[ "${VERL_HOME}" == /* ]] || { echo "ERROR: VERL_HOME must be an absolute path" >&2; exit 2; }
DATA_HOME="${VERL_HOME}"
DATA_DIR="${DATA_DIR:-${DATA_HOME}/data/eval}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_HOME}/models}"
EVAL_OUTPUTS="${EVAL_OUTPUTS:-${DATA_HOME}/eval_outputs}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/.venv}"
[[ "${UV_PROJECT_ENVIRONMENT}" == /* ]] || UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}"
export PROJECT_ROOT VERL_HOME DATA_HOME DATA_DIR MODEL_ROOT EVAL_OUTPUTS UV_PROJECT_ENVIRONMENT

# All model families use the same uv project and absolute Python executable.
# The former LEGACY_EVAL_PYTHON / CUDA_MODULE_LEGACY entries in existing .env
# files are intentionally ignored; users do NOT need to edit their .env.
EVAL_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
EVAL_PROFILE="${EVAL_PROFILE:?Submit through a model-specific eval_*.sh}"
case "${EVAL_PROFILE}" in
    llmjp) MATH_EVAL_SCRIPT="${PROJECT_ROOT}/llmjp_eval.py" ;;
    qwen) MATH_EVAL_SCRIPT="${PROJECT_ROOT}/qwen_eval.py" ;;
    phi|dsdt) MATH_EVAL_SCRIPT="${PROJECT_ROOT}/math_eval.py" ;;
    *) echo "ERROR: unknown EVAL_PROFILE=${EVAL_PROFILE}" >&2; exit 2 ;;
esac
CUDA_MODULE="${SYNKA_CUDA_MODULE:-cuda/12.8}"
export HF_HOME="${HF_HOME:-${DATA_HOME}/cache/huggingface}"
export EVAL_PYTHON MATH_EVAL_SCRIPT CUDA_MODULE

SYNKA_PROJECT_ID="${SYNKA_PROJECT_ID:-}"
SYNKA_PARTITION="${SYNKA_PARTITION:-${SYNKA_PROJECT_ID}}"
SYNKA_NODELIST="${SYNKA_NODELIST:-}"
SYNKA_SUBMIT_HOST="${SYNKA_SUBMIT_HOST:-cpusv06}"
SYNKA_GPU_TYPE="${SYNKA_GPU_TYPE:-H200}"
SYNKA_JOB_PREFIX="${SYNKA_JOB_PREFIX:-eval}"
SYNKA_MEM_PER_GPU_GB="${SYNKA_MEM_PER_GPU_GB:-32}"
MAX_TASK_RETRIES="${MAX_TASK_RETRIES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-15}"
LOCK_WAIT_SECONDS="${LOCK_WAIT_SECONDS:-90}"

synka_die() {
    echo "ERROR: $*" >&2
    exit 2
}

synka_load_modules() {
    if ! command -v module >/dev/null 2>&1; then
        for init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            if [[ -r "${init}" ]]; then
                # shellcheck disable=SC1090
                source "${init}"
                break
            fi
        done
    fi
    command -v module >/dev/null 2>&1 || synka_die "module command is unavailable on $(hostname)"
    # Load the toolkit module available on SYNKA. Validate vLLM/Torch on H200 separately.
    module purge
    module load "${CUDA_MODULE}"
}

synka_sanitize_job_component() {
    local value="$1"
    value="${value//[^A-Za-z0-9_-]/-}"
    printf '%s' "${value:0:48}"
}

synka_prepare_dataset_cache() {
    local dataset="$1"
    local split="$2"
    local cache_file="${DATA_DIR}/${dataset}/${split}.jsonl"
    [[ -f "${cache_file}" ]] && return 0

    echo "Preparing ${dataset} -> ${cache_file}"
    (
        cd "${PROJECT_ROOT}"
        "${EVAL_PYTHON}" - "${dataset}" "${split}" "${DATA_DIR}" <<'PY'
from data_loader import load_data
import sys
name, split, data_dir = sys.argv[1:]
rows = load_data(name, split, data_dir)
print(f"Prepared {name}: {len(rows)} examples")
PY
    )
}

synka_prepare_shared_caches() {
    local data_names="$1"
    local split="$2"
    case ",${data_names}," in *",gpqa_diamond,"*) synka_prepare_dataset_cache gpqa_diamond "${split}" ;; esac
    case ",${data_names}," in *",mmlu_stem,"*) synka_prepare_dataset_cache mmlu_stem "${split}" ;; esac
    case ",${data_names}," in *",livecodebench_v6,"*) synka_prepare_dataset_cache livecodebench_v6 "${split}" ;; esac
    # Prebuild all requested transfer datasets on cpusv06,
    # never concurrently on several GPU nodes.
    local requested
    for requested in gsm_plus gsm_plus_seed gsm_symbolic_main gsm_symbolic_p1 gsm_symbolic_p2 \
        math_perturb_simple math_perturb_hard omega_exploratory \
        omega_compositional omega_transformative processbench_gsm8k \
        processbench_math processbench_olympiadbench proofbench obligation_transfer; do
        case ",${data_names}," in *",${requested},"*) synka_prepare_dataset_cache "${requested}" "${split}" ;; esac
    done
}

synka_write_owner_json() {
    local owner_file="$1"
    local exp_name="$2"
    local attempt="$3"
    "${EVAL_PYTHON}" - "${owner_file}" "${exp_name}" "${attempt}" <<'PY'
import json, os, socket, sys, time
path, exp_name, attempt = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = {
    "experiment": exp_name,
    "hostname": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "attempt": attempt,
    "pid": os.getpid(),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
}

synka_worker() {
    local exp_name=""
    local data_names=""
    local sample_num="16"
    local split="test"
    local num_test_sample="-1"
    local prompt_type="qwen25-math-cot"
    local max_tokens_per_call="38912"
    local seed="42"
    local temperature="0.6"
    local top_p="0.95"
    local top_k="20"
    local min_p="0.0"
    local presence_penalty="0.0"
    local start="0"
    local end="-1"
    local pipeline_parallel_size="1"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --exp) exp_name="$2"; shift 2 ;;
            --data-names) data_names="$2"; shift 2 ;;
            --sample-num) sample_num="$2"; shift 2 ;;
            --split) split="$2"; shift 2 ;;
            --num-test-sample) num_test_sample="$2"; shift 2 ;;
            --prompt-type) prompt_type="$2"; shift 2 ;;
            --max-tokens-per-call) max_tokens_per_call="$2"; shift 2 ;;
            --seed) seed="$2"; shift 2 ;;
            --temperature) temperature="$2"; shift 2 ;;
            --top-p) top_p="$2"; shift 2 ;;
            --top-k) top_k="$2"; shift 2 ;;
            --min-p) min_p="$2"; shift 2 ;;
            --presence-penalty) presence_penalty="$2"; shift 2 ;;
            --start) start="$2"; shift 2 ;;
            --end) end="$2"; shift 2 ;;
            --pipeline-parallel-size) pipeline_parallel_size="$2"; shift 2 ;;
            *) synka_die "unknown worker argument: $1" ;;
        esac
    done

    [[ -n "${exp_name}" ]] || synka_die "worker requires --exp"
    [[ -n "${data_names}" ]] || synka_die "worker requires --data-names"

    # Do not rely on the Slurm launch directory or its spooled script path.
    cd -- "${PROJECT_ROOT}"
    synka_load_modules
    [[ -x "${EVAL_PYTHON}" ]] || synka_die "unified uv Python is missing: ${EVAL_PYTHON} (run uv sync)"
    [[ -f "${MATH_EVAL_SCRIPT}" ]] || synka_die "evaluation script is missing: ${MATH_EVAL_SCRIPT}"

    local model_path="${MODEL_ROOT}/${exp_name}"
    local output_dir="${EVAL_OUTPUTS}/${exp_name}_${sample_num}"
    local eval_log="${output_dir}/evaluation.log"
    local lock_file="${output_dir}/.evaluation.lock"
    local owner_file="${output_dir}/.evaluation_running.json"

    [[ -d "${model_path}" ]] || synka_die "model directory is missing: ${model_path}"
    mkdir -p "${output_dir}"
    [[ "${EVAL_PROFILE}" == llmjp ]] && chmod 700 "${output_dir}"

    exec 9>"${lock_file}"
    if ! flock -w "${LOCK_WAIT_SECONDS}" 9; then
        echo "ERROR: another evaluation is already writing ${output_dir}" >&2
        [[ -f "${owner_file}" ]] && cat "${owner_file}" >&2 || true
        exit 75
    fi

    cleanup_owner() { rm -f -- "${owner_file}"; }
    trap cleanup_owner EXIT INT TERM HUP

    export RAY_DATA_HOME="${DATA_HOME}"
    export TOKENIZERS_PARALLELISM=false

    echo "======================================================"
    echo "SYNKA evaluation worker"
    echo "  experiment:       ${exp_name}"
    echo "  slurm job:        ${SLURM_JOB_ID:-not-set}"
    echo "  host:             $(hostname)"
    echo "  CUDA module:      ${CUDA_MODULE}"
    echo "  CUDA_VISIBLE:     ${CUDA_VISIBLE_DEVICES:-not-set}"
    echo "  model:            ${model_path}"
    echo "  output:           ${output_dir}"
    echo "======================================================"
    nvidia-smi || true

    local cmd=(
        "${EVAL_PYTHON}" -u "${MATH_EVAL_SCRIPT}"
        --model_name_or_path "${model_path}"
        --data_dir "${DATA_DIR}"
        --data_names "${data_names}"
        --output_dir "${output_dir}"
        --split "${split}"
        --prompt_type "${prompt_type}"
        --num_test_sample "${num_test_sample}"
        --max_tokens_per_call "${max_tokens_per_call}"
        --seed "${seed}"
        --temperature "${temperature}"
        --n_sampling "${sample_num}"
        --top_p "${top_p}"
        --top_k "${top_k}"
        --start "${start}"
        --end "${end}"
        --pipeline_parallel_size "${pipeline_parallel_size}"
    )

    if [[ "${EVAL_PROFILE}" == phi || "${EVAL_PROFILE}" == dsdt ]]; then
        cmd+=(--use_vllm --apply_chat_template --save_outputs)
    fi

    if [[ "${EVAL_PROFILE}" == llmjp ]]; then
        cmd+=(--reasoning_effort "${LLMJP_REASONING_EFFORT:-high}"
              --question_batch_size "${LLMJP_QUESTION_BATCH_SIZE:-4}"
              --max_model_len "${LLMJP_MAX_MODEL_LEN:-65536}")
    elif [[ "${EVAL_PROFILE}" == qwen ]]; then
        cmd+=(--min_p "${min_p}" --presence_penalty "${presence_penalty}"
              --question_batch_size "${QWEN_QUESTION_BATCH_SIZE:-4}"
              --max_model_len "${QWEN_MAX_MODEL_LEN:-40960}")
    fi

    local max_attempts=$((MAX_TASK_RETRIES + 1))
    local attempt rc=1
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        synka_write_owner_json "${owner_file}" "${exp_name}" "${attempt}"
        {
            echo
            echo "======================================================"
            echo "[START] experiment=${exp_name} attempt=${attempt}/${max_attempts}"
            echo "time=$(date --iso-8601=seconds)"
            printf 'COMMAND:'
            printf ' %q' "${cmd[@]}"
            echo
            echo "======================================================"
        } | tee -a "${eval_log}"

        set +e
        "${cmd[@]}" 2>&1 | tee -a "${eval_log}"
        rc=${PIPESTATUS[0]}
        set -e

        if [[ ${rc} -eq 0 ]]; then
            echo "[SUCCEEDED] ${exp_name}" | tee -a "${eval_log}"
            cleanup_owner
            trap - EXIT INT TERM HUP
            return 0
        fi

        echo "[FAILED] ${exp_name} attempt=${attempt}/${max_attempts} rc=${rc}" | tee -a "${eval_log}" >&2
        (( attempt < max_attempts )) && sleep "${RETRY_DELAY_SECONDS}"
    done

    cleanup_owner
    trap - EXIT INT TERM HUP
    return "${rc}"
}

# Public helper used by the three model-specific submitters.
# Required globals set by caller:
#   EXP_NAMES DATA_NAMES SAMPLE_NUM SPLIT NUM_TEST_SAMPLE PROMPT_TYPE
#   MAX_TOKENS_PER_CALL SEED TEMPERATURE TOP_P TOP_K START END
#   PIPELINE_PARALLEL_SIZE GPUS_PER_EXP CPUS_PER_EXP
# Site-specific safety: every job uses one of exactly eight explicitly allowed nodes.
synka_validate_cluster() {
    [[ "${SYNKA_PROJECT_ID}" && "${SYNKA_PROJECT_ID}" != *REPLACE* ]] || synka_die "set SYNKA_PROJECT_ID in .env_synka"
    [[ "${SYNKA_PARTITION}" && "${SYNKA_PARTITION}" != *REPLACE* ]] || synka_die "set SYNKA_PARTITION in .env_synka"
    [[ "${SYNKA_NODELIST}" && "${SYNKA_NODELIST}" != *REPLACE* ]] || synka_die "set the allocated eight SYNKA_NODELIST nodes in .env_synka"
    [[ "${SYNKA_GPU_TYPE}" == "H200" ]] || synka_die "SYNKA guide specifies GPU type H200"
    [[ "$(hostname -s)" == "${SYNKA_SUBMIT_HOST}" ]] || synka_die "submit on ${SYNKA_SUBMIT_HOST}, not $(hostname -s)"
    command -v scontrol >/dev/null || synka_die "scontrol is required"
    command -v sinfo >/dev/null || synka_die "sinfo is required"
    local allowed_raw partition_raw
    allowed_raw="$(scontrol show hostnames "${SYNKA_NODELIST}")" || synka_die "invalid SYNKA_NODELIST"
    mapfile -t SYNKA_ALLOWED_NODES < <(printf '%s\n' "${allowed_raw}" | sed '/^$/d' | sort -u)
    (( ${#SYNKA_ALLOWED_NODES[@]} == 8 )) || synka_die "SYNKA_NODELIST must resolve to exactly 8 distinct nodes (got ${#SYNKA_ALLOWED_NODES[@]})"
    local host
    for host in "${SYNKA_ALLOWED_NODES[@]}"; do
        [[ "${host}" =~ ^gpusv[0-9]{2}$ ]] || synka_die "unexpected SYNKA GPU host: ${host}"
    done
    partition_raw="$(sinfo -a -h -N -p "${SYNKA_PARTITION}" -o '%N')" || synka_die "cannot inspect SYNKA_PARTITION"
    [[ -n "${partition_raw}" ]] || synka_die "partition has no visible GPU nodes: ${SYNKA_PARTITION}"
    local -A members=()
    while IFS= read -r host; do
        [[ -z "${host}" ]] || members["${host}"]=1
    done <<< "${partition_raw}"
    for host in "${SYNKA_ALLOWED_NODES[@]}"; do
        [[ -n "${members[${host}]+yes}" ]] || synka_die "node ${host} is not in partition ${SYNKA_PARTITION}"
    done
}

synka_submit_experiments() {
    local caller_script="$1"
    [[ -x "${EVAL_PYTHON}" ]] || synka_die "unified uv Python is missing: ${EVAL_PYTHON} (run uv sync)"
    [[ -f "${MATH_EVAL_SCRIPT}" ]] || synka_die "evaluation script is missing: ${MATH_EVAL_SCRIPT}"
    command -v sbatch >/dev/null || synka_die "sbatch unavailable; submit from cpusv06"
    [[ "${GPUS_PER_EXP}" =~ ^[1-8]$ ]] || synka_die "GPUS_PER_EXP must be 1..8 (one H200 node)"
    [[ "${CPUS_PER_EXP}" =~ ^[1-9][0-9]*$ ]] || synka_die "CPUS_PER_EXP must be positive"
    (( CPUS_PER_EXP <= 112 )) || synka_die "H200 node has 112 CPU cores"
    [[ "${SYNKA_MEM_PER_GPU_GB}" =~ ^[1-9][0-9]*$ ]] || synka_die "SYNKA_MEM_PER_GPU_GB must be positive"
    local mem_gb=$(( GPUS_PER_EXP * SYNKA_MEM_PER_GPU_GB ))
    (( mem_gb <= 2048 )) || synka_die "SYNKA H200 node has 2048 GiB memory"
    (( ${#EXP_NAMES[@]} > 0 )) || synka_die "EXP_NAMES is empty"
    synka_validate_cluster

    # Project-dedicated partition can request unlimited walltime. Shared is capped at 24 h.
    local synka_walltime
    if [[ "${SYNKA_PARTITION}" == "ungrouped" ]]; then
        synka_walltime="${SYNKA_WALLTIME:-12:00:00}"
        [[ "${synka_walltime}" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]] || synka_die "shared partition requires HH:MM:SS walltime (<=24h)"
        local seconds=$(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
        (( seconds > 0 && seconds <= 86400 )) || synka_die "shared partition max time is 24 hours"
    else
        [[ "${SYNKA_PARTITION}" == "${SYNKA_PROJECT_ID}" ]] || synka_die "dedicated partition must match SYNKA_PROJECT_ID"
        synka_walltime="${SYNKA_WALLTIME:-0}"
    fi

    declare -A seen=()
    local exp_name
    for exp_name in "${EXP_NAMES[@]}"; do
        [[ -z "${seen[${exp_name}]+x}" ]] || synka_die "duplicate experiment: ${exp_name}"
        seen["${exp_name}"]=1
        [[ -d "${MODEL_ROOT}/${exp_name}" ]] || synka_die "model directory is missing: ${MODEL_ROOT}/${exp_name}"
    done
    mkdir -p "${EVAL_OUTPUTS}"
    [[ "${EVAL_PROFILE}" == llmjp ]] && chmod 700 "${EVAL_OUTPUTS}"
    synka_prepare_shared_caches "${DATA_NAMES}" "${SPLIT}"

    local run_id submission_dir log_dir manifest
    run_id="$(date +%Y%m%d_%H%M%S_%N)_$$_${RANDOM}"
    submission_dir="${EVAL_OUTPUTS}/synka_submissions/${run_id}"
    log_dir="${submission_dir}/logs"
    manifest="${submission_dir}/jobs.tsv"
    mkdir -p "${log_dir}"
    printf 'experiment\tjob_id\tjob_name\tlog_pattern\n' > "${manifest}"

    local -a sbatch_base=(
        sbatch --parsable
        --partition="${SYNKA_PARTITION}"
        --account="${SYNKA_PROJECT_ID}"
        --nodelist="${SYNKA_NODELIST}"
        --nodes=1
        --gres="gpu:${SYNKA_GPU_TYPE}:${GPUS_PER_EXP}"
        --cpus-per-task="${CPUS_PER_EXP}"
        --mem="${mem_gb}G"
        --time="${synka_walltime}"
        --chdir="${PROJECT_ROOT}"
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},EVAL_ENV_FILE=${ENV_FILE},EVAL_PROFILE=${EVAL_PROFILE}"
    )
    echo "Submitting ${#EXP_NAMES[@]} SYNKA ${EVAL_PROFILE} job(s) on only ${SYNKA_NODELIST}"
    echo "Partition=${SYNKA_PARTITION}, account=${SYNKA_PROJECT_ID}, time=${synka_walltime}, log=${log_dir}"
    local -a job_ids=()
    local idx=0 exp_tag job_name log_pattern job_id
    for exp_name in "${EXP_NAMES[@]}"; do
        idx=$((idx + 1))
        exp_tag="$(synka_sanitize_job_component "${exp_name}")"
        job_name="${SYNKA_JOB_PREFIX}-${idx}-${exp_tag}"
        job_name="${job_name:0:96}"
        log_pattern="${log_dir}/${job_name}_%j.out"
        job_id="$(
            "${sbatch_base[@]}" \
                --job-name="${job_name}" \
                --output="${log_pattern}" \
                --error="${log_pattern}" \
                "${SCRIPT_DIR}/_eval_synka_common.sh" __worker \
                --exp "${exp_name}" \
                --data-names "${DATA_NAMES}" \
                --sample-num "${SAMPLE_NUM}" \
                --split "${SPLIT}" \
                --num-test-sample "${NUM_TEST_SAMPLE}" \
                --prompt-type "${PROMPT_TYPE}" \
                --max-tokens-per-call "${MAX_TOKENS_PER_CALL}" \
                --seed "${SEED}" \
                --temperature "${TEMPERATURE}" \
                --top-p "${TOP_P}" \
                --top-k "${TOP_K}" \
                --start "${START}" \
                --end "${END}" \
                --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
        )" || synka_die "sbatch submission failed for ${exp_name}"
        job_id="${job_id%%;*}"
        job_ids+=("${job_id}")
        printf '%s\t%s\t%s\t%s\n' "${exp_name}" "${job_id}" "${job_name}" "${log_pattern}" >> "${manifest}"
        echo "Submitted ${exp_name} -> ${job_id}"
    done
    local job_csv
    job_csv="$(IFS=,; echo "${job_ids[*]}")"
    echo "Status: squeue -j ${job_csv}"
    echo "Manifest: ${manifest}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "${1:-}" == "__worker" ]]; then
        shift
        synka_worker "$@"
    else
        synka_die "_eval_synka_common.sh is internal; run a model-specific eval_*.sh"
    fi
fi
