#!/usr/bin/env bash
# Share the same gitignored local settings without depending on MDX internals.
_NICT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
_NICT_ENV_FILE="${EVAL_ENV_FILE:-${_NICT_PROJECT_ROOT}/.env}"
if [[ -r "${_NICT_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${_NICT_ENV_FILE}"
    set +a
fi
export PROJECT_ROOT="${PROJECT_ROOT:-${_NICT_PROJECT_ROOT}}"

UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/.venv}"
[[ "${UV_PROJECT_ENVIRONMENT}" == /* ]] || UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}"
EVAL_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
export UV_PROJECT_ENVIRONMENT EVAL_PYTHON
