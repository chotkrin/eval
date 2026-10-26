#!/usr/bin/env bash
# Direct (non-sandboxed) scoring of previously generated LiveCodeBench outputs
# using the official LiveCodeBench evaluator.
# WARNING: this executes model-generated Python code with the current user's permissions.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_local_config.sh"

LCB_ROOT="${LCB_ROOT:?Set LCB_ROOT in .env}"
LCB_PYTHON="${LCB_PYTHON:-${LCB_ROOT}/.venv/bin/python}"
RELEASE_VERSION="${LCB_RELEASE_VERSION:-v6}"
CUSTOM_OUTPUT_FILE="${1:?usage: score_livecodebench.sh /path/to/*_lcb_custom.json}"

if [[ ! -x "${LCB_PYTHON}" ]]; then
    echo "ERROR: LiveCodeBench Python not found: ${LCB_PYTHON}" >&2
    exit 2
fi
if [[ ! -d "${LCB_ROOT}/lcb_runner" ]]; then
    echo "ERROR: LiveCodeBench checkout not found: ${LCB_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${CUSTOM_OUTPUT_FILE}" ]]; then
    echo "ERROR: custom output not found: ${CUSTOM_OUTPUT_FILE}" >&2
    exit 2
fi

cat >&2 <<'WARN'
WARNING: DIRECT MODE. The official LiveCodeBench evaluator will execute
model-generated programs with the permissions of the current user.
WARN

cd "${LCB_ROOT}"
exec "${LCB_PYTHON}" -m lcb_runner.runner.custom_evaluator \
    --scenario codegeneration \
    --release_version "${RELEASE_VERSION}" \
    --custom_output_file "${CUSTOM_OUTPUT_FILE}" \
    --n "${LCB_N:-16}" \
    --temperature "${LCB_TEMPERATURE:-0.6}" \
    --num_process_evaluate "${LCB_NUM_PROCESSES:-64}" \
    --timeout "${LCB_TIMEOUT:-6}"
