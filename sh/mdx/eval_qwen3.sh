#!/usr/bin/env bash
# Official-model-baseline evaluation. Edit the model lists below.
# Both use the same uv Python as LLM-jp/Phi/DSDT and write flat dataset results.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export EVAL_PROFILE=qwen
source "${SCRIPT_DIR}/_eval_mdx_common.sh"

SAMPLE_NUM=16
SPLIT="test"
NUM_TEST_SAMPLE=-1
DATA_NAMES="math500,aime24,minerva_math,olympiadbench,amc23,aime26,gpqa_diamond,mmlu_stem,livecodebench_v6"
SEED=42
TOP_P=0.95
TOP_K=20
MIN_P=0.0
START=0
END=-1
PIPELINE_PARALLEL_SIZE=1
GPUS_PER_EXP=4
CPUS_PER_EXP=32

# These are directory names below MODEL_ROOT, not Hugging Face repo IDs.
QWEN3_EXP_NAMES=(
  "Qwen3-8B"
)
QWEN35_EXP_NAMES=(
  "Qwen3.5-9B"
)

# Qwen3 native vLLM config exposes 40,960 total context tokens.
# Official recommended maximum output is 32,768; 38,912 is a separate
# long-context benchmarking configuration requiring explicit YaRN.
if (( ${#QWEN3_EXP_NAMES[@]} > 0 )); then
    EXP_NAMES=("${QWEN3_EXP_NAMES[@]}")
    PROMPT_TYPE="qwen3-thinking"
    TEMPERATURE=0.6
    PRESENCE_PENALTY=0.0
    MAX_TOKENS_PER_CALL=32768
    QWEN_MAX_MODEL_LEN=40960
    export QWEN_MAX_MODEL_LEN
    mdx_submit_experiments "${BASH_SOURCE[0]}:Qwen3"
fi

# Qwen3.5 native context is much longer; use 65,536 total tokens for
# comparable eval budgets. Official general-thinking: temperature=1.0,
# top_p=0.95, top_k=20, presence_penalty=1.5. For precise coding tasks,
# official recommendation is temp=0.6 / presence_penalty=0.0; run as a
# separate dataset-only configuration when you want that protocol.
if (( ${#QWEN35_EXP_NAMES[@]} > 0 )); then
    EXP_NAMES=("${QWEN35_EXP_NAMES[@]}")
    PROMPT_TYPE="qwen35-thinking"
    TEMPERATURE=1.0
    PRESENCE_PENALTY=1.5
    MAX_TOKENS_PER_CALL=38912
    QWEN_MAX_MODEL_LEN=65536
    export QWEN_MAX_MODEL_LEN
    mdx_submit_experiments "${BASH_SOURCE[0]}:Qwen3.5"
fi
