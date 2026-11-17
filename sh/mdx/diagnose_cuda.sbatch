#!/usr/bin/env bash
#SBATCH --job-name=0001_eval-8020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=cuda-compare-%j.out
#SBATCH --error=cuda-compare-%j.err

# Run from the project root (contains .env):
#   sbatch --job-name=0001_eval-8020 --nodelist=llm-jp-noble-gpu10 sh/mdx/diagnose_cuda.sbatch
# The same allocated GPU is tested with: inherited env, module purge,
# and the exact CUDA module loaded by sh/mdx/_eval_mdx_common.sh.
set -uo pipefail

ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -r "${EVAL_ENV_FILE:-${ROOT}/.env}" ]]; then
    source "${EVAL_ENV_FILE:-${ROOT}/.env}"
fi
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT:-${ROOT}/.venv}/bin/python"
MODULE_NAME="${EVAL_CUDA_MODULE:-${CUDA_MODULE_LLMJP:-cuda/13.0/13.0.3}}"

printf '\n=== Node and allocation ===\n'
hostname
date -Is
printf 'job=%s node=%s visible=%s allocated=%s python=%s module=%s\n' \
  "${SLURM_JOB_ID:-?}" "${SLURM_JOB_NODELIST:-?}" \
  "${CUDA_VISIBLE_DEVICES:-UNSET}" "${SLURM_JOB_GPUS:-UNSET}" \
  "$PYTHON_BIN" "$MODULE_NAME"
command -v nvidia-smi >/dev/null && nvidia-smi -L || true
printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-UNSET}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python missing: $PYTHON_BIN"
  exit 2
fi

probe() {
    local stage="$1"
    printf '\n=== CUDA PROBE: %s ===\n' "$stage"
    printf 'CUDA_VISIBLE_DEVICES=%s\nLD_LIBRARY_PATH=%s\n' \
        "${CUDA_VISIBLE_DEVICES:-UNSET}" "${LD_LIBRARY_PATH:-UNSET}"
    if "$PYTHON_BIN" - <<'PY'
import os, sys, traceback
try:
    import torch
    print('python:', sys.executable, flush=True)
    print('torch:', torch.__version__, 'wheel_cuda:', torch.version.cuda, flush=True)
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
    torch.cuda.init()
    n = torch.cuda.device_count()
    print('CUDA INIT PASS, count:', n, flush=True)
    for i in range(n):
        x = torch.ones(1, device=f'cuda:{i}')
        torch.cuda.synchronize(i)
        print('device:', i, torch.cuda.get_device_name(i), 'compute:', x.item(), flush=True)
except BaseException as e:
    print('CUDA PROBE FAIL:', type(e).__name__, str(e), flush=True)
    traceback.print_exc()
    sys.exit(1)
PY
    then
      printf '[RESULT] %s PASS\n' "$stage"
    else
      printf '[RESULT] %s FAIL\n' "$stage"
    fi
}

probe inherited

if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$init" ]]; then source "$init"; break; fi
    done
fi
if command -v module >/dev/null 2>&1; then
    if module purge; then
        probe purged
    else
        echo '[RESULT] module purge failed'
    fi
    if [[ "${MODULE_NAME,,}" == none || "${MODULE_NAME,,}" == off || "${MODULE_NAME,,}" == skip ]]; then
        echo '[RESULT] explicit CUDA module loading disabled; current profile skips module changes'
    elif module load "$MODULE_NAME"; then
        probe loaded
    else
        echo "[RESULT] module load failed: $MODULE_NAME"
    fi
else
    echo '[RESULT] module command unavailable; only inherited environment tested'
fi
