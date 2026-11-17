#!/usr/bin/env bash
# Three-level summary of completed eval metrics. Edit EXP_NAMES here.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
if [[ -r "${EVAL_ENV_FILE:-${PROJECT_ROOT}/.env}" ]]; then
  set -a
  source "${EVAL_ENV_FILE:-${PROJECT_ROOT}/.env}"
  set +a
fi
EVAL_OUTPUTS="${EVAL_OUTPUTS:?Set EVAL_OUTPUTS in .env}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/.venv}"
[[ "$UV_PROJECT_ENVIRONMENT" == /* ]] || UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}"
EVAL_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
LCB_N="${LCB_N:-16}"
[[ -x "$EVAL_PYTHON" ]] || { echo "Missing uv Python: $EVAL_PYTHON" >&2; exit 2; }

# Edit this list in THIS script; the scoring and summary lists are independent.
EXP_NAMES=(
  "exp004_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-Overlong4K"
  "sci-llm-jp-4-8b-sft"
  # "Phi-4B-0"
  # "DAPO-DSDT-1.5B-LASER-L1024-a1.25"
)

SUMMARY_CSV="${SUMMARY_CSV:-${EVAL_OUTPUTS}/eval_summary_n${LCB_N}.csv}"
"$EVAL_PYTHON" "$SCRIPT_DIR/lcb_results.py" summary \
  --output-dir "$EVAL_OUTPUTS" --n "$LCB_N" \
  --csv "$SUMMARY_CSV" "${EXP_NAMES[@]}"
