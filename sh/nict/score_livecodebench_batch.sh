#!/usr/bin/env bash
# Batch-score LiveCodeBench generations under eval_outputs/<experiment>/livecodebench_v6/.
#
# Usage:
#   bash sh/nict/score_livecodebench_batch.sh
#   bash sh/nict/score_livecodebench_batch.sh 'DAPO-DSDT-1\.5B-.*'
#   bash sh/nict/score_livecodebench_batch.sh '.*LEASH.*'
#
# Environment overrides:
#   EVAL_OUTPUTS=<your-output-directory>
#   LCB_ROOT=${LCB_ROOT}
#   LCB_PYTHON=${LCB_ROOT}/.venv/bin/python
#   LCB_SCORER=/path/to/score_livecodebench.sh
#   LCB_RELEASE_VERSION=v6
#   LCB_N=16
#   LCB_TEMPERATURE=0.6
#   LCB_NUM_PROCESSES=4
#   LCB_TIMEOUT=6
#   LCB_FORCE_RESCORE=1   # rescore even when complete official score files exist

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_config.sh"
EVAL_OUTPUTS="${EVAL_OUTPUTS:?Set EVAL_OUTPUTS in .env}"
LCB_ROOT="${LCB_ROOT:?Set LCB_ROOT in .env}"
LCB_PYTHON="${LCB_PYTHON:-${LCB_ROOT}/.venv/bin/python}"
LCB_SCORER="${LCB_SCORER:-${SCRIPT_DIR}/score_livecodebench.sh}"
LCB_FORCE_RESCORE="${LCB_FORCE_RESCORE:-0}"

# The official custom evaluator writes these three companion files for
# <name>_lcb_custom.json:
#   <name>_lcb_custom_codegeneration_output.json
#   <name>_lcb_custom_codegeneration_output_eval.json
#   <name>_lcb_custom_codegeneration_output_eval_all.json
# A score is considered complete only if all three are non-empty, parse as JSON,
# and the per-problem output/eval_all lengths match the custom input.
is_complete_score() {
    local custom_file="$1"
    local output_file eval_file eval_all_file

    output_file="${custom_file%.json}_codegeneration_output.json"
    eval_file="${output_file%.json}_eval.json"
    eval_all_file="${output_file%.json}_eval_all.json"

    [[ -s "${output_file}" && -s "${eval_file}" && -s "${eval_all_file}" ]] || return 1

    "${LCB_PYTHON}" - "${custom_file}" "${output_file}" "${eval_file}" "${eval_all_file}" <<'PY' >/dev/null 2>&1
import json
import sys

custom_path, output_path, eval_path, eval_all_path = sys.argv[1:]
try:
    with open(custom_path, "r", encoding="utf-8") as f:
        custom = json.load(f)
    with open(output_path, "r", encoding="utf-8") as f:
        output = json.load(f)
    with open(eval_path, "r", encoding="utf-8") as f:
        json.load(f)
    with open(eval_all_path, "r", encoding="utf-8") as f:
        eval_all = json.load(f)
except (OSError, json.JSONDecodeError, UnicodeDecodeError):
    raise SystemExit(1)

if not isinstance(custom, list) or not isinstance(output, list) or not isinstance(eval_all, list):
    raise SystemExit(1)
if len(custom) == 0:
    raise SystemExit(1)
if len(output) != len(custom) or len(eval_all) != len(custom):
    raise SystemExit(1)
PY
}

# Extended regular expression matched against the experiment directory name.
# "*" is accepted as an alias for no filtering.
EXP_REGEX="${1:-.*}"
if [[ "${EXP_REGEX}" == "*" ]]; then
    EXP_REGEX='.*'
fi

if [[ ! -d "${EVAL_OUTPUTS}" ]]; then
    echo "ERROR: eval outputs directory not found: ${EVAL_OUTPUTS}" >&2
    exit 2
fi
if [[ ! -f "${LCB_SCORER}" ]]; then
    echo "ERROR: scorer not found: ${LCB_SCORER}" >&2
    exit 2
fi
if ! printf '%s\n' test | grep -Eq "${EXP_REGEX}" 2>/dev/null && \
   ! printf '%s\n' '' | grep -Eq "${EXP_REGEX}" 2>/dev/null; then
    # Distinguish a valid non-matching regex from an invalid regex.
    if grep -Eq "${EXP_REGEX}" </dev/null 2>/dev/null; then
        :
    else
        rc=$?
        if [[ ${rc} -eq 2 ]]; then
            echo "ERROR: invalid extended regex: ${EXP_REGEX}" >&2
            exit 2
        fi
    fi
fi

echo "============================================================"
echo "LiveCodeBench batch scoring"
echo "EVAL_OUTPUTS : ${EVAL_OUTPUTS}"
echo "EXP_REGEX    : ${EXP_REGEX}"
echo "LCB_ROOT     : ${LCB_ROOT}"
echo "LCB_PYTHON   : ${LCB_PYTHON}"
echo "LCB_SCORER   : ${LCB_SCORER}"
echo "FORCE_RESCORE: ${LCB_FORCE_RESCORE}"
echo "============================================================"

matched=0
succeeded=0
skipped=0
failed=0

# NUL-delimited traversal keeps experiment/file names safe.
while IFS= read -r -d '' custom_file; do
    lcb_dir="$(dirname -- "${custom_file}")"
    exp_dir="$(dirname -- "${lcb_dir}")"
    exp_name="$(basename -- "${exp_dir}")"

    if ! printf '%s\n' "${exp_name}" | grep -Eq "${EXP_REGEX}"; then
        continue
    fi

    ((matched += 1))
    echo
    echo "[${matched}] ${exp_name}"
    echo "    ${custom_file}"

    if [[ "${LCB_FORCE_RESCORE}" != "1" ]] && is_complete_score "${custom_file}"; then
        ((skipped += 1))
        echo "[SKIP scored] ${exp_name}"
        continue
    fi

    if LCB_ROOT="${LCB_ROOT}" \
       LCB_PYTHON="${LCB_PYTHON}" \
       bash "${LCB_SCORER}" "${custom_file}" </dev/null; then
        ((succeeded += 1))
        echo "[OK]   ${exp_name}"
    else
        rc=$?
        ((failed += 1))
        echo "[FAIL] ${exp_name} (exit=${rc})" >&2
    fi
done < <(
    find "${EVAL_OUTPUTS}" \
        -type f \
        -path '*/livecodebench_v6/*_lcb_custom.json' \
        -print0 \
        | sort -z
)

echo
echo "============================================================"
echo "Batch summary"
echo "Matched   : ${matched}"
echo "Skipped   : ${skipped}"
echo "Succeeded : ${succeeded}"
echo "Failed    : ${failed}"
echo "============================================================"

if [[ ${matched} -eq 0 ]]; then
    echo "No LiveCodeBench custom outputs matched regex: ${EXP_REGEX}" >&2
    exit 0
fi

if [[ ${failed} -ne 0 ]]; then
    exit 1
fi
