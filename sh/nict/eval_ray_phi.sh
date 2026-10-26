#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_local_config.sh"

export RAY_DATA_HOME="${RAY_DATA_HOME:?Set RAY_DATA_HOME in .env}"
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}"

# Single uv-managed interpreter for Ray driver AND model evaluation.
# The Ray cluster must have a matching Ray/Python version.
RAY_PYTHON="${EVAL_PYTHON}"
RAY_CLI="${UV_PROJECT_ENVIRONMENT}/bin/ray"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
DISPATCH_SCRIPT="eval_dispatch.py"
MATH_EVAL_SCRIPT="math_eval.py"

SAMPLE_NUM=16
SPLIT="test"
NUM_TEST_SAMPLE=-1
PROMPT_TYPE="phi4-mini-reasoning"
DATA_DIR="${RAY_DATA_HOME:?Set RAY_DATA_HOME in .env}/data/eval"
# DATA_NAMES="math500,aime24,minerva_math,olympiadbench,amc23,aime26,gpqa_diamond,mmlu_stem,livecodebench_v6"
DATA_NAMES="gpqa_diamond,mmlu_stem,livecodebench_v6"

# Each experiment independently requests these resources from Ray.
GPUS_PER_EXP=8
CPUS_PER_EXP=16

# Retries are only for Ray/node failures and duplicate-output lock conflicts.
MAX_TASK_RETRIES=3
RETRY_DELAY_SECONDS=15

EXP_NAMES=(
  # "Phi-4-mini-reasoning"
  # "Phi-4B-unsup-5e-5_10"
  # "Phi-4B-unsup-5e-5_20"
  # "Phi-4B-unsup-5e-5_30"
  # "Phi-4B-unsup-5e-5_40"
  # "Phi-4B-unsup-5e-5_50"
  # "Phi-4B-unsup-5e-5_60"
  # "Phi-4B-unsup-5e-5_70"
  # "Phi-4B-unsup-5e-5_80"
  # "Phi-4B-unsup-5e-5_90"
  # "Phi-4B-unsup-5e-5_100"
  # "Phi-4B-unsup-5e-5_110"
  # "Phi-4B-unsup-5e-5_120"
  "Phi-4B-5e-5"
  "Phi-4B-5e-4"
  "Phi-4B-1e-4"
  "Phi-4B-1e-3"
  "Phi-4B-3e-4"
  "Phi-4B-7e-4"
  # "Phi-4B-1e-2"
  # "Phi-4B-5e-3"
  # "Phi-4B-3e-3"
)

for executable in "${RAY_PYTHON}" "${RAY_CLI}" "${EVAL_PYTHON}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "ERROR: executable is missing: ${executable}" >&2
        exit 2
    fi
done

for required in \
    "${PROJECT_ROOT}/${DISPATCH_SCRIPT}" \
    "${PROJECT_ROOT}/${MATH_EVAL_SCRIPT}"
do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: required file is missing: ${required}" >&2
        exit 2
    fi
done

if (( ${#EXP_NAMES[@]} == 0 )); then
    echo "ERROR: EXP_NAMES is empty." >&2
    exit 2
fi

"${RAY_PYTHON}" - <<'PY'
import ray
import sys
print(
    f"Ray driver: Python={sys.version.split()[0]} "
    f"Ray={ray.__version__} executable={sys.executable}"
)
PY

EXP_ARGS=()
for exp_name in "${EXP_NAMES[@]}"; do
    EXP_ARGS+=(--exp "${exp_name}")
done

# Upload code only. Models, datasets, resume files, and outputs stay on /fast.
RUNTIME_ENV_JSON="$("${RAY_PYTHON}" - <<PY
import json
print(json.dumps({
    "working_dir": "${PROJECT_ROOT}",
    "excludes": [
        "/.git/",
        "/.git/**",
        # Never send local credentials or uv wheels to Ray's working_dir.
        "/.env",
        "/.env.*",
        "/.venv/",
        "/.venv/**",
        "**/__pycache__/",
        "**/*.pyc",
        "/DAPO-DSDT-*/",
        "/Phi-4-mini-reasoning_*/",
        "/DeepSeek-R1-Distill-Qwen-1.5B_*/",
        "/eval_outputs/",
    ],
}))
PY
)"

# Include nanoseconds, PID, and RANDOM so rapid repeated submissions stay unique.
RUN_ID="$(date +%Y%m%d_%H%M%S_%N)_$$_${RANDOM}"
SUBMISSION_ID="eval_queue_${RUN_ID}"
STATUS_FILE="${RAY_DATA_HOME}/eval_outputs/${SUBMISSION_ID}_status.json"

echo "======================================================"
echo "Submitting simple Ray evaluation coordinator"
echo "  submission-id:    ${SUBMISSION_ID}"
echo "  experiments:      ${#EXP_NAMES[@]}"
echo "  GPUs/experiment:  ${GPUS_PER_EXP}"
echo "  CPUs/experiment:  ${CPUS_PER_EXP}"
echo "  total GPU reserve: none"
echo "  status file:      ${STATUS_FILE}"
echo "======================================================"

"${RAY_CLI}" job submit \
    --address "${RAY_API_SERVER_ADDRESS}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --submission-id "${SUBMISSION_ID}" \
    --entrypoint-num-cpus 1 \
    --no-wait \
    -- \
    env \
        RAY_DATA_HOME="${RAY_DATA_HOME}" \
        TOKENIZERS_PARALLELISM=false \
    "${RAY_PYTHON}" "${DISPATCH_SCRIPT}" \
        --run-id "${RUN_ID}" \
        --status-file "${STATUS_FILE}" \
        --python-executable "${EVAL_PYTHON}" \
        --math-eval-script "${MATH_EVAL_SCRIPT}" \
        --ray-data-home "${RAY_DATA_HOME}" \
        --data-dir "${DATA_DIR}" \
        --data-names "${DATA_NAMES}" \
        --sample-num "${SAMPLE_NUM}" \
        --split "${SPLIT}" \
        --num-test-sample "${NUM_TEST_SAMPLE}" \
        --prompt-type "${PROMPT_TYPE}" \
        --gpus-per-exp "${GPUS_PER_EXP}" \
        --cpus-per-exp "${CPUS_PER_EXP}" \
        --max-task-retries "${MAX_TASK_RETRIES}" \
        --retry-delay-seconds "${RETRY_DELAY_SECONDS}" \
        --temperature 0.8 \
        "${EXP_ARGS[@]}"

echo
echo "Submitted: ${SUBMISSION_ID}"
echo
echo "Coordinator log:"
echo "  ${RAY_CLI} job logs -f ${SUBMISSION_ID} --address ${RAY_API_SERVER_ADDRESS}"
echo
echo "Structured coordinator status:"
echo "  watch -n 3 'jq . \"${STATUS_FILE}\"'"
echo
echo "Actual Ray task status across all Jobs:"
echo "  ${RAY_CLI} list tasks --detail --filter state=RUNNING"
echo "  ${RAY_CLI} list tasks --detail --filter state=PENDING_NODE_ASSIGNMENT"
echo "  ${RAY_CLI} summary tasks"
