#!/usr/bin/env bash
# Flatten legacy LLM-jp result folders into each dataset directory. Edit EXP_NAMES below.
# Dry-run: bash analysis_results/migrate.sh
# Apply:   bash analysis_results/migrate.sh --apply
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

# Edit this list for the experiments to migrate.
EXP_NAMES=(
  "exp001_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-Correct"
  "exp002_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-LEASH4K"
  "exp003_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-LASERD4K"
  "exp004_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-Overlong4K"
  "sci-llm-jp-4-8b-sft"
  "ELYZA-Thinking-1.0-llm-jp-4-32b-a3b"
  "dapo-sci-sft-rl100k-enja-v3-e0a-tis-step360"
)
N_SAMPLING=16
"$EVAL_PYTHON" "$SCRIPT_DIR/migrate.py" \
  --output-dir "$EVAL_OUTPUTS" --n "$N_SAMPLING" "$@" "${EXP_NAMES[@]}"
