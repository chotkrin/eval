#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${EVAL_ENV_FILE:-${ROOT}/.env}"
if [[ -r "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${ROOT}/.venv}"
[[ "${UV_PROJECT_ENVIRONMENT}" == /* ]] || UV_PROJECT_ENVIRONMENT="${ROOT}/${UV_PROJECT_ENVIRONMENT}"
PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing shared uv environment: ${PYTHON}" >&2; exit 2; }
"${PYTHON}" - <<'PYCODE'
import sys
from importlib.metadata import version
assert sys.version_info[:3] == (3, 13, 5), sys.version
for name, expected in [("vllm", "0.30.0"), ("llm-jp-vllm", "0.1.1"), ("ray", "2.56.1")]:
    actual = version(name)
    print(f"{name}={actual}")
    assert actual == expected, f"{name}: expected {expected}, got {actual}"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from llm_jp_vllm.llmjp4.harmony import HarmonyMessageParser
from latex2sympy2 import latex2sympy
from sympy import Symbol
assert latex2sympy(r"x") == Symbol("x")
for module in ("math_eval", "llmjp_eval"):
    __import__(module)
print("Unified Phi/DSDT/LLM-jp imports: OK", sys.executable)
PYCODE
"${PYTHON}" -m unittest discover -s "${ROOT}/tests" -v
