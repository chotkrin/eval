#!/usr/bin/env bash
# Run official LCB correctness tests ONLY for missing/incomplete score files.
# Edit EXP_NAMES below to select experiments. The generated code is untrusted;
# run inside an isolated CPU container/VM without secrets or sensitive mounts.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
if [[ -r "${EVAL_ENV_FILE:-${PROJECT_ROOT}/.env}" ]]; then
  set -a
  source "${EVAL_ENV_FILE:-${PROJECT_ROOT}/.env}"
  set +a
fi
EVAL_OUTPUTS="${EVAL_OUTPUTS:?Set EVAL_OUTPUTS in .env}"
LCB_ROOT="${LCB_ROOT:?Set LCB_ROOT in .env}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/.venv}"
[[ "$UV_PROJECT_ENVIRONMENT" == /* ]] || UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}"
EVAL_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
LCB_N="${LCB_N:-16}"
LCB_RELEASE_VERSION="${LCB_RELEASE_VERSION:-v6}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-0.6}"
LCB_NUM_PROCESSES="${LCB_NUM_PROCESSES:-4}"
LCB_TIMEOUT="${LCB_TIMEOUT:-6}"
LCB_FORCE_RESCORE="${LCB_FORCE_RESCORE:-0}"

# Edit this list in THIS script. Same names as sh/mdx/eval_llmjp.sh.
EXP_NAMES=(
  "exp001_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-Correct"
  "exp002_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-LEASH4K"
  "exp003_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-LASERD4K"
  "exp004_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-Overlong4K"
  "sci-llm-jp-4-8b-sft"
  "ELYZA-Thinking-1.0-llm-jp-4-32b-a3b"
  "dapo-sci-sft-rl100k-enja-v3-e0a-tis-step360"
  "Qwen3-8B"
)

[[ -x "$EVAL_PYTHON" ]] || { echo "Missing uv Python: $EVAL_PYTHON" >&2; exit 2; }
[[ -d "$LCB_ROOT/lcb_runner" ]] || { echo "Missing LCB checkout: $LCB_ROOT" >&2; exit 2; }
[[ "$LCB_RELEASE_VERSION" == v6 ]] || { echo "Only incremental LCB v6 is supported for livecodebench_v6" >&2; exit 2; }
# Explicit consent: the LCB runner executes arbitrary programs from the model.
[[ "${LCB_ACK_UNTRUSTED_CODE:-0}" == 1 ]] || {
  echo 'ERROR: LCB executes untrusted model code. Use an isolated no-network CPU VM/container' >&2
  echo 'with no secrets, then explicitly set LCB_ACK_UNTRUSTED_CODE=1.' >&2
  exit 2
}
# Do not give the official runner access to HF_TOKEN or similar secrets.
UNSET_SECRET_VARS=(HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN)
failed=0; skipped=0; completed=0; missing=0
for exp in "${EXP_NAMES[@]}"; do
  target="${EVAL_OUTPUTS}/${exp}_${LCB_N}/livecodebench_v6"
  if [[ ! -d "$target" ]]; then
    echo "[MISSING] ${exp}: $target" >&2
    ((missing+=1))
    continue
  fi
  # Use the same run selector as analysis and migration; recursive find can
  # otherwise disagree when historical hash folders still exist.
  if ! custom="$("$EVAL_PYTHON" "$SCRIPT_DIR/lcb_results.py" select-custom "$target" --n "$LCB_N")"; then
    echo "[ERROR] ${exp}: cannot select a unique valid custom input" >&2
    ((failed+=1))
    continue
  fi
  if [[ "$LCB_FORCE_RESCORE" != 1 ]] && "$EVAL_PYTHON" "$SCRIPT_DIR/lcb_results.py" score-status "$custom" --n "$LCB_N"; then
    ((skipped+=1))
    continue
  fi
  echo "[SCORING] $exp -> $custom"
  log="${custom%.json}_codegeneration_evaluation.log"
  # One scoring job per experiment; never change the upstream generation files.
  if (
    for key in "${UNSET_SECRET_VARS[@]}"; do unset "$key"; done
    export PYTHONPATH="$LCB_ROOT"
    cd "$LCB_ROOT"
    "$EVAL_PYTHON" -m lcb_runner.runner.custom_evaluator \
      --scenario codegeneration \
      --release_version "$LCB_RELEASE_VERSION" \
      --custom_output_file "$custom" \
      --n "$LCB_N" \
      --temperature "$LCB_TEMPERATURE" \
      --num_process_evaluate "$LCB_NUM_PROCESSES" \
      --timeout "$LCB_TIMEOUT"
  ) >"$log" 2>&1 && "$EVAL_PYTHON" "$SCRIPT_DIR/lcb_results.py" score-status "$custom" --n "$LCB_N"; then
    echo "[DONE] $exp ($log)"
    ((completed+=1))
  else
    echo "[FAILED] $exp; see $log" >&2
    ((failed+=1))
  fi
done
printf 'LCB scoring: completed=%s skipped=%s missing=%s failed=%s\n' "$completed" "$skipped" "$missing" "$failed"
(( failed == 0 && missing == 0 ))
