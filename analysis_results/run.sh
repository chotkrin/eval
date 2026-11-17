#!/usr/bin/env bash
# Read-only analysis of existing MDX evaluations. Edit EXP_NAMES below.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
if [[ -r "${EVAL_ENV_FILE:-${PROJECT_ROOT}/.env}" ]]; then
  set -a
  source "${EVAL_ENV_FILE:-${PROJECT_ROOT}/.env}"
  set +a
fi
EVAL_OUTPUTS="${EVAL_OUTPUTS:?Set EVAL_OUTPUTS in .env}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/.venv}"
[[ "$UV_PROJECT_ENVIRONMENT" == /* ]] || UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}"
EVAL_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
[[ -x "$EVAL_PYTHON" ]] || { echo "Missing uv Python: $EVAL_PYTHON" >&2; exit 2; }

# Edit these lists in this shell. Names must match the experiment output folders.
# Exactly the four uncommented EXP_NAMES from sh/mdx/eval.sh.
# Keep names verbatim: output folders are EVAL_OUTPUTS/<name>_<N_SAMPLING>/.
EXP_NAMES=(
  "DAPO-DSDT-1.5B-0-3072"
  "DAPO-DSDT-1.5B-0"
  "DAPO-DSDT-1.5B-LASER-D-LB4096-a0.5"
  "DAPO-DSDT-1.5B-LEASH-Lt4096-llr0.005"
  "DeepSeek-R1-Distill-Qwen-1.5B"
)
# Same comma-separated selection semantics as DATA_NAMES in sh/mdx/eval.sh.
# These SEVEN names are exactly the uncommented DATA_NAMES from that file.
# Edit DATASETS to analyze other already-run datasets; order is preserved.
# For one-off overrides: ANALYSIS_DATASETS='math500,processbench_gsm8k' bash analysis_results/run.sh
# Compatible extra names: all Math/MC/Code datasets, GSM-Plus, GSM-Symbolic,
# MATH-Perturb, OMEGA, ProcessBench, ProofBench and obligation_transfer.
DATASETS="${ANALYSIS_DATASETS:-gsm_plus,gsm_plus_seed,gsm_symbolic_main,gsm_symbolic_p1,gsm_symbolic_p2,math_perturb_simple,math_perturb_hard}"
BASELINE="DeepSeek-R1-Distill-Qwen-1.5B"
N_SAMPLING=16
PASS_K=(1 5 10)

# Store analysis outputs inside this directory; reports/ is gitignored.
REPORT_DIR="${ANALYSIS_REPORT_DIR:-${SCRIPT_DIR}/reports_selected}"
K_LIST="$(IFS=,; echo "${PASS_K[*]}")"
"$EVAL_PYTHON" "$SCRIPT_DIR/analyze.py" \
  --output-dir "$EVAL_OUTPUTS" --report-dir "$REPORT_DIR" \
  --n "$N_SAMPLING" --ks "$K_LIST" --datasets "$DATASETS" \
  --baseline "$BASELINE" --allow-unavailable-truncation "${EXP_NAMES[@]}"
