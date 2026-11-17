#!/usr/bin/env bash
# Download and normalize public evaluation-only benchmarks.
# Run on an internet-connected login node, not on an allocated GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

: "${DATA_DIR:?Set DATA_DIR in your project .env}"
: "${UV_PROJECT_ENVIRONMENT:?Set UV_PROJECT_ENVIRONMENT in your project .env}"
[[ "${DATA_DIR}" = /* ]] || { echo 'DATA_DIR must be absolute' >&2; exit 2; }
[[ "${UV_PROJECT_ENVIRONMENT}" = /* ]] || { echo 'UV_PROJECT_ENVIRONMENT must be absolute' >&2; exit 2; }
PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing uv Python: ${PYTHON}" >&2; exit 2; }

# Editable list. Do NOT automatically mix these into the original nine-dataset
# headline: ProcessBench and ProofBench require different scorers.
DATASETS=(
  gsm_plus gsm_plus_seed
  gsm_symbolic_main gsm_symbolic_p1 gsm_symbolic_p2
  math_perturb_simple math_perturb_hard
  processbench_gsm8k processbench_math processbench_olympiadbench processbench_omnimath
  proofbench
  prmbench_error_steps proofflowbench_graph proofflowbench_proof
  # omega_exploratory omega_compositional omega_transformative
)

# The default raw folder is the same one used by transfer_datasets.py.
RAW_DIR="${TRANSFER_RAW_DIR:-$(dirname "${DATA_DIR}")/raw}"
export TRANSFER_RAW_DIR="${RAW_DIR}"
mkdir -p "${DATA_DIR}" "${RAW_DIR}"

clone_once() {
  local url="$1" dest="$2"
  if [[ -d "${dest}/.git" ]]; then
    echo "[EXISTS] ${dest} (not automatically updated; pin/review upstream revision)"
  elif [[ -e "${dest}" ]]; then
    echo "ERROR: ${dest} exists but is not a Git checkout" >&2
    return 2
  else
    git clone --depth 1 "${url}" "${dest}"
  fi
  git -C "${dest}" rev-parse HEAD
}

# Clone the official OMEGA generator as part of the source download. Creating
# a reproducible fixed test set is a separate step and requires validated exports.
DOWNLOAD_OMEGA_SOURCE=1
need_math=0 need_omega=0
for name in "${DATASETS[@]}"; do
  case "${name}" in
    math_perturb_simple|math_perturb_hard) need_math=1 ;;
    omega_exploratory|omega_compositional|omega_transformative) need_omega=1 ;;
  esac
done
if (( need_math )); then
  clone_once https://github.com/Kaffaljidhmah2/MATH-Perturb.git "${RAW_DIR}/MATH-Perturb"
fi
if (( need_omega || DOWNLOAD_OMEGA_SOURCE )); then
  clone_once https://github.com/sunblaze-ucb/omega.git "${RAW_DIR}/omega"
  echo 'OMEGA is a generator, not a fixed downloadable test set.'
  echo "Prepare and verify ${RAW_DIR}/omega/exports/omega_<axis>.jsonl first."
  echo 'See TRANSFER_DATASETS.md for required fields and generation procedure.'
fi

cd "${PROJECT_ROOT}"
for name in "${DATASETS[@]}"; do
  target="${DATA_DIR}/${name}/test.jsonl"
  if [[ -s "${target}" ]]; then
    echo "[EXISTS] ${target}"
    continue
  fi
  echo "[PREPARE] ${name}"
  "${PYTHON}" transfer_prepare.py --data-dir "${DATA_DIR}" --datasets "${name}"
  [[ -s "${target}" ]] || { echo "ERROR: no normalized output: ${target}" >&2; exit 2; }
done

echo '[DONE] Normalized datasets are under:' "${DATA_DIR}/<dataset>/test.jsonl"
