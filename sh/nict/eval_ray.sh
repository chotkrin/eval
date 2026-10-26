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
PROMPT_TYPE="qwen25-math-cot"
DATA_DIR="${RAY_DATA_HOME:?Set RAY_DATA_HOME in .env}/data/eval"
# DATA_NAMES="math500,aime24,minerva_math,olympiadbench,amc23,aime26,gpqa_diamond,mmlu_stem,livecodebench_v6"
DATA_NAMES="math500,aime24,minerva_math,olympiadbench,amc23,aime26,gpqa_diamond,mmlu_stem,livecodebench_v6"

# Each experiment independently requests these resources from Ray.
GPUS_PER_EXP=4
CPUS_PER_EXP=16

# Retries are only for Ray/node failures and duplicate-output lock conflicts.
MAX_TASK_RETRIES=3
RETRY_DELAY_SECONDS=15

EXP_NAMES=(
  # "DAPO-DSDT-1.5B-smtm-1.5e-4"
  # "DAPO-DSDT-1.5B-smtm-3e-4"
  # "DAPO-DSDT-1.5B-smtm-5e-5"
  # "DAPO-DSDT-1.5B-LASER-L1024-a0.5"
  # "DAPO-DSDT-1.5B-LASER-L2048-a0.5"
  # "DAPO-DSDT-1.5B-LASER-L3072-a0.5"
  # "DAPO-DSDT-1.5B-LASER-L4096-a0.5"
  # "DAPO-DSDT-1.5B-1.5e-4"
  # "DAPO-DSDT-1.5B-1e-3"
  # "DAPO-DSDT-1.5B-1e-4"
  # "DAPO-DSDT-1.5B-1e-5"
  # "DAPO-DSDT-1.5B-2e-5"
  # "DAPO-DSDT-1.5B-3e-4"
  # "DAPO-DSDT-1.5B-3e-5"
  # "DAPO-DSDT-1.5B-4e-5"
  # "DAPO-DSDT-1.5B-5e-3"
  # "DAPO-DSDT-1.5B-5e-4"
  # "DAPO-DSDT-1.5B-5e-5"
  # "DAPO-DSDT-1.5B-6e-5"
  # "DAPO-DSDT-1.5B-7e-5"
  # "DAPO-DSDT-1.5B-8e-5"
  # "DAPO-DSDT-1.5B-9e-5"
  # "DAPO-DSDT-1.5B-LEASH-Lt1024-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt128-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt1536-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt2048-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt256-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt4096-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt512-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt6144-llr0.005"
  # "DAPO-DSDT-1.5B-unsup-5e-5_10"
  # "DAPO-DSDT-1.5B-unsup-5e-5_15"
  # "DAPO-DSDT-1.5B-unsup-5e-5_20"
  # "DAPO-DSDT-1.5B-unsup-5e-5_25"
  # "DAPO-DSDT-1.5B-unsup-5e-5_30"
  # "DAPO-DSDT-1.5B-unsup-5e-5_35"
  # "DAPO-DSDT-1.5B-unsup-5e-5_40"
  # "DAPO-DSDT-1.5B-unsup-5e-5_45"
  # "DAPO-DSDT-1.5B-unsup-5e-5_5"
  # "DAPO-DSDT-1.5B-unsup-5e-5_50"
  # "DAPO-DSDT-1.5B-unsup-5e-5_55"
  # "DAPO-DSDT-1.5B-unsup-5e-5_60"
  # "DAPO-DSDT-1.5B-unsup-5e-5_65"
  # "DAPO-DSDT-1.5B-unsup-5e-5_70"
  # "DAPO-DSDT-1.5B-unsup-5e-5_75"
  # "DAPO-DSDT-1.5B-unsup-5e-5_80"
  # "DAPO-DSDT-1.5B-unsup-5e-5_85"
  # "DAPO-DSDT-1.5B-unsup-5e-5_90"
  # "DAPO-DSDT-7B-unsup-5e-5_10
  # "DAPO-DSDT-7B-unsup-5e-5_20
  # "DAPO-DSDT-7B-unsup-5e-5_30
  # "DAPO-DSDT-7B-unsup-5e-5_40
  # "DAPO-DSDT-7B-unsup-5e-5_45"
  # "DAPO-DSDT-7B-unsup-5e-5_50"
  # "DAPO-DSDT-7B-unsup-5e-5_55"
  # "DAPO-DSDT-7B-unsup-5e-5_60"
  # "DAPO-DSDT-7B-unsup-5e-5_65"
  # "DAPO-DSDT-7B-unsup-5e-5_70"
  # "DAPO-DSDT-7B-unsup-5e-5_75"
  # "DAPO-DSDT-7B-1.5e-4"
  # "DAPO-DSDT-7B-1e-3"
  # "DAPO-DSDT-7B-1e-4"
  # "DAPO-DSDT-7B-1e-5"
  # "DAPO-DSDT-7B-5e-3"
  # "DAPO-DSDT-7B-5e-4"
  # "DAPO-DSDT-7B-3e-4"
  # "DAPO-DSDT-7B-5e-5"
  "DAPO-DSDT-7B-3e-5"
  # "DeepSeek-R1-Distill-Qwen-1.5B"
  # "DeepSeek-R1-Distill-Qwen-7B"
  # "DAPO-DSDT-1.5B-unsup-5e-5_10"
  # "DAPO-DSDT-1.5B-unsup-5e-5_45"
  # "DAPO-DSDT-1.5B-unsup-5e-5_55"
  # "DAPO-DSDT-1.5B-unsup-5e-5_20"
  # "DAPO-DSDT-1.5B-unsup-5e-5_30"
  # "DAPO-DSDT-1.5B-unsup-5e-5_40"
  # "DAPO-DSDT-1.5B-unsup-5e-5_50"
  # "DAPO-DSDT-1.5B-unsup-5e-5_60"
  # "DAPO-DSDT-1.5B-unsup-5e-5_65"
  # "DAPO-DSDT-1.5B-unsup-5e-5_70"
  # "DAPO-DSDT-1.5B-unsup-5e-5_75"
  # "DAPO-DSDT-1.5B-unsup-5e-5_80"
  # "DAPO-DSDT-1.5B-unsup-5e-5_85"
  # "DAPO-DSDT-1.5B-unsup-5e-5_90"
  # "DAPO-DSDT-1.5B-LEASH-Lt4096-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt6144-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt2048-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt1536-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt1024-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt512-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt256-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt128-llr0.005"
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

# GPQA Diamond is gated and its official Hugging Face split is named `train`.
# Normalize it once on the submission node before launching concurrent Ray tasks.
# This avoids multiple experiments racing to create the same shared test.jsonl.
if [[ ",${DATA_NAMES}," == *",gpqa_diamond,"* ]]; then
    GPQA_CACHE_FILE="${DATA_DIR}/gpqa_diamond/${SPLIT}.jsonl"
    if [[ ! -f "${GPQA_CACHE_FILE}" ]]; then
        echo "Preparing GPQA Diamond -> ${GPQA_CACHE_FILE}"
        (
            cd "${PROJECT_ROOT}"
            "${EVAL_PYTHON}" - <<PY
from data_loader import load_data
rows = load_data("gpqa_diamond", "${SPLIT}", "${DATA_DIR}")
print(f"Prepared GPQA Diamond: {len(rows)} examples")
PY
        )
    fi
fi



# MMLU-STEM is public. Normalize/cache it once on the submission node so
# concurrent Ray experiments do not all download/filter/write the same file.
if [[ ",${DATA_NAMES}," == *",mmlu_stem,"* ]]; then
    MMLU_STEM_CACHE_FILE="${DATA_DIR}/mmlu_stem/${SPLIT}.jsonl"
    if [[ ! -f "${MMLU_STEM_CACHE_FILE}" ]]; then
        echo "Preparing MMLU-STEM -> ${MMLU_STEM_CACHE_FILE}"
        (
            cd "${PROJECT_ROOT}"
            "${EVAL_PYTHON}" - <<PY
from data_loader import load_data
rows = load_data("mmlu_stem", "${SPLIT}", "${DATA_DIR}")
print(f"Prepared MMLU-STEM: {len(rows)} examples")
PY
        )
    fi
fi

# LiveCodeBench generation data is public. Cache the explicit v6 increment once
# before concurrent Ray tasks. No generated program is executed here.
if [[ ",${DATA_NAMES}," == *",livecodebench_v6,"* ]]; then
    LCB_CACHE_FILE="${DATA_DIR}/livecodebench_v6/${SPLIT}.jsonl"
    if [[ ! -f "${LCB_CACHE_FILE}" ]]; then
        echo "Preparing LiveCodeBench v6 -> ${LCB_CACHE_FILE}"
        (
            cd "${PROJECT_ROOT}"
            "${EVAL_PYTHON}" - <<PY
from data_loader import load_data
rows = load_data("livecodebench_v6", "${SPLIT}", "${DATA_DIR}")
print(f"Prepared LiveCodeBench v6: {len(rows)} examples")
PY
        )
    fi
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
