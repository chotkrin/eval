#!/usr/bin/env bash
# Shared MDX/Slurm implementation for eval.sh and eval_phi.sh.
# Do not edit experiment lists here; edit EXP_NAMES in the model-specific submitter.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_config.sh"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

# MDX storage root requested for this project.
DATA_HOME="${DATA_HOME:?Set DATA_HOME in .env}"
DATA_DIR="${DATA_DIR:-${DATA_HOME}/data/eval}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_HOME}/models}"
EVAL_OUTPUTS="${EVAL_OUTPUTS:-${DATA_HOME}/eval_outputs}"

# Shared uv evaluator; loaded from _local_config.sh.
EVAL_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
MATH_EVAL_SCRIPT="${MATH_EVAL_SCRIPT:-${PROJECT_ROOT}/math_eval.py}"

CUDA_MODULE="${EVAL_CUDA_MODULE:-${CUDA_MODULE_LLMJP:-cuda/13.0/13.0.3}}"
MDX_PARTITION="${MDX_PARTITION:-gpu}"
MDX_QOS="${MDX_QOS:-}"
MDX_WALLTIME="${MDX_WALLTIME:-20:00:00}"
MDX_JOB_PREFIX="${MDX_JOB_PREFIX:-0001_eval}"
MDX_MEM_PER_GPU_GB="${MDX_MEM_PER_GPU_GB:-32}"
MAX_TASK_RETRIES="${MAX_TASK_RETRIES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-15}"
LOCK_WAIT_SECONDS="${LOCK_WAIT_SECONDS:-90}"

mdx_die() {
    echo "ERROR: $*" >&2
    exit 2
}

mdx_load_modules() {
    if ! command -v module >/dev/null 2>&1; then
        for init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            if [[ -r "${init}" ]]; then
                # shellcheck disable=SC1090
                source "${init}"
                break
            fi
        done
    fi
    command -v module >/dev/null 2>&1 || mdx_die "module command is unavailable on $(hostname)"
    module load "${CUDA_MODULE}"
}

mdx_sanitize_job_component() {
    local value="$1"
    value="${value//[^A-Za-z0-9_-]/-}"
    printf '%s' "${value:0:48}"
}

mdx_prepare_dataset_cache() {
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

mdx_prepare_shared_caches() {
    local data_names="$1"
    local split="$2"
    case ",${data_names}," in *",gpqa_diamond,"*) mdx_prepare_dataset_cache gpqa_diamond "${split}" ;; esac
    case ",${data_names}," in *",mmlu_stem,"*) mdx_prepare_dataset_cache mmlu_stem "${split}" ;; esac
    case ",${data_names}," in *",livecodebench_v6,"*) mdx_prepare_dataset_cache livecodebench_v6 "${split}" ;; esac
}

mdx_write_owner_json() {
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

mdx_worker() {
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
            --start) start="$2"; shift 2 ;;
            --end) end="$2"; shift 2 ;;
            --pipeline-parallel-size) pipeline_parallel_size="$2"; shift 2 ;;
            *) mdx_die "unknown worker argument: $1" ;;
        esac
    done

    [[ -n "${exp_name}" ]] || mdx_die "worker requires --exp"
    [[ -n "${data_names}" ]] || mdx_die "worker requires --data-names"

    mdx_load_modules
    [[ -x "${EVAL_PYTHON}" ]] || mdx_die "evaluation Python is missing: ${EVAL_PYTHON}"
    [[ -f "${MATH_EVAL_SCRIPT}" ]] || mdx_die "math_eval.py is missing: ${MATH_EVAL_SCRIPT}"

    local model_path="${MODEL_ROOT}/${exp_name}"
    local output_dir="${EVAL_OUTPUTS}/${exp_name}_${sample_num}"
    local eval_log="${output_dir}/evaluation.log"
    local lock_file="${output_dir}/.evaluation.lock"
    local owner_file="${output_dir}/.evaluation_running.json"

    [[ -d "${model_path}" ]] || mdx_die "model directory is missing: ${model_path}"
    mkdir -p "${output_dir}"

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
    echo "MDX evaluation worker"
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
        --use_vllm
        --apply_chat_template
        --save_outputs
    )

    local max_attempts=$((MAX_TASK_RETRIES + 1))
    local attempt rc=1
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        mdx_write_owner_json "${owner_file}" "${exp_name}" "${attempt}"
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

# Public helper used by the two model-specific submitters.
# Required globals set by caller:
#   EXP_NAMES DATA_NAMES SAMPLE_NUM SPLIT NUM_TEST_SAMPLE PROMPT_TYPE
#   MAX_TOKENS_PER_CALL SEED TEMPERATURE TOP_P TOP_K START END
#   PIPELINE_PARALLEL_SIZE GPUS_PER_EXP CPUS_PER_EXP
mdx_submit_experiments() {
    local caller_script="$1"

    [[ -x "${EVAL_PYTHON}" ]] || mdx_die "evaluation Python is missing: ${EVAL_PYTHON}"
    [[ -f "${MATH_EVAL_SCRIPT}" ]] || mdx_die "math_eval.py is missing: ${MATH_EVAL_SCRIPT}"
    [[ -f "${PROJECT_ROOT}/data_loader.py" ]] || mdx_die "data_loader.py is missing: ${PROJECT_ROOT}/data_loader.py"
    command -v sbatch >/dev/null 2>&1 || mdx_die "sbatch is unavailable; submit from login.llm-jp.mdx.jp"

    [[ "${GPUS_PER_EXP}" =~ ^[1-9][0-9]*$ ]] || mdx_die "GPUS_PER_EXP must be a positive integer"
    [[ "${CPUS_PER_EXP}" =~ ^[1-9][0-9]*$ ]] || mdx_die "CPUS_PER_EXP must be a positive integer"
    [[ "${MDX_MEM_PER_GPU_GB}" =~ ^[1-9][0-9]*$ ]] || mdx_die "MDX_MEM_PER_GPU_GB must be a positive integer"
    (( CPUS_PER_EXP <= 16 * GPUS_PER_EXP )) || mdx_die "MDX rule: CPU request exceeds 16 cores/GPU"
    (( MDX_MEM_PER_GPU_GB <= 48 )) || mdx_die "MDX rule: memory request exceeds 48 GiB/GPU"
    (( ${#EXP_NAMES[@]} > 0 )) || mdx_die "EXP_NAMES is empty"

    declare -A seen=()
    local exp_name
    for exp_name in "${EXP_NAMES[@]}"; do
        [[ -z "${seen[${exp_name}]+x}" ]] || mdx_die "duplicate experiment: ${exp_name}"
        seen["${exp_name}"]=1
        [[ -d "${MODEL_ROOT}/${exp_name}" ]] || mdx_die "model directory is missing: ${MODEL_ROOT}/${exp_name}"
    done

    mkdir -p "${EVAL_OUTPUTS}"
    mdx_prepare_shared_caches "${DATA_NAMES}" "${SPLIT}"

    local run_id submission_dir log_dir manifest mem_gb
    run_id="$(date +%Y%m%d_%H%M%S_%N)_$$_${RANDOM}"
    submission_dir="${EVAL_OUTPUTS}/mdx_submissions/${run_id}"
    log_dir="${submission_dir}/logs"
    manifest="${submission_dir}/jobs.tsv"
    mem_gb=$((GPUS_PER_EXP * MDX_MEM_PER_GPU_GB))
    mkdir -p "${log_dir}"
    printf 'experiment\tjob_id\tjob_name\tlog_pattern\n' > "${manifest}"

    local -a sbatch_base=(
        sbatch --parsable
        --partition="${MDX_PARTITION}"
        --gres="gpu:${GPUS_PER_EXP}"
        --cpus-per-task="${CPUS_PER_EXP}"
        --mem="${mem_gb}G"
        --time="${MDX_WALLTIME}"
        --requeue
    )
    [[ -n "${MDX_QOS}" ]] && sbatch_base+=(--qos="${MDX_QOS}")

    echo "======================================================"
    echo "Submitting MDX Slurm evaluations"
    echo "  submitter:        ${caller_script}"
    echo "  experiments:      ${#EXP_NAMES[@]}"
    echo "  GPUs/experiment:  ${GPUS_PER_EXP}"
    echo "  CPUs/experiment:  ${CPUS_PER_EXP}"
    echo "  memory/job:       ${mem_gb}G"
    echo "  data names:       ${DATA_NAMES}"
    echo "  manifest:         ${manifest}"
    echo "======================================================"

    local -a job_ids=()
    local idx=0 job_name exp_tag log_pattern job_id
    for exp_name in "${EXP_NAMES[@]}"; do
        idx=$((idx + 1))
        exp_tag="$(mdx_sanitize_job_component "${exp_name}")"
        job_name="${MDX_JOB_PREFIX}-${idx}-${exp_tag}"
        job_name="${job_name:0:96}"
        log_pattern="${log_dir}/${job_name}_%j.out"

        job_id="$(
            "${sbatch_base[@]}" \
                --job-name="${job_name}" \
                --output="${log_pattern}" \
                --error="${log_pattern}" \
                "${SCRIPT_DIR}/_eval_mdx_common.sh" __worker \
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
        )"
        job_id="${job_id%%;*}"
        job_ids+=("${job_id}")
        printf '%s\t%s\t%s\t%s\n' "${exp_name}" "${job_id}" "${job_name}" "${log_pattern}" >> "${manifest}"
        echo "Submitted ${exp_name} -> job ${job_id} (${job_name})"
    done

    local job_csv
    job_csv="$(IFS=,; echo "${job_ids[*]}")"
    echo
    echo "Submitted ${#job_ids[@]} job(s)."
    echo "Status:   squeue -j ${job_csv}"
    echo "Manifest: ${manifest}"
    echo "Logs:     ${log_dir}"
    echo "Cancel:   scancel ${job_ids[*]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "${1:-}" == "__worker" ]]; then
        shift
        mdx_worker "$@"
    else
        mdx_die "_eval_mdx_common.sh is internal; run eval.sh or eval_phi.sh"
    fi
fi
