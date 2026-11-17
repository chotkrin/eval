#!/usr/bin/env bash
# Download only peer-reviewed, obligation-adjacent public benchmark data.
# Safe to run on a SLURM login node; no model loading or GPU allocation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ -r "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi
: "${DATA_HOME:?Set DATA_HOME in .env or environment}"
DATA_DIR="${DATA_DIR:-${DATA_HOME}/data/eval}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/.venv}"
case "${UV_PROJECT_ENVIRONMENT}" in
  /*) ;;
  *) UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}" ;;
esac
[[ "${DATA_DIR}" = /* ]] || { echo 'DATA_DIR must be absolute' >&2; exit 2; }
PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing uv Python: ${PYTHON}" >&2; exit 2; }
export HF_HOME="${HF_HOME:-${DATA_HOME}/cache/huggingface}"
mkdir -p "${DATA_DIR}" "${HF_HOME}"

# New tasks; edit this list to omit ones you don't intend to evaluate.
DATASETS=(
  proofflowbench_graph
  proofflowbench_proof
  prmbench_error_steps
  processbench_omnimath
)

cd "${PROJECT_ROOT}"
for dataset in "${DATASETS[@]}"; do
  file="${DATA_DIR}/${dataset}/test.jsonl"
  if [[ -s "${file}" ]]; then
    echo "[EXISTS] ${dataset}: ${file}"
    continue
  fi
  echo "[PREPARE] ${dataset}"
  "${PYTHON}" transfer_prepare.py --data-dir "${DATA_DIR}" --datasets "${dataset}"
  [[ -s "${file}" ]] || { echo "Missing/empty ${file}" >&2; exit 2; }
done

echo "[DONE] Data files saved to ${DATA_DIR}/<dataset>/test.jsonl"
echo '[NOTE] ProofFlowBench and PRMBench publish only train splits; local test.jsonl is for evaluation convenience, not an official held-out split.'
