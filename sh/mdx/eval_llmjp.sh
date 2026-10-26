#!/usr/bin/env bash
# MDX/Slurm equivalent of sh/nict/eval_ray.sh.
# Edit EXP_NAMES below; every enabled entry is submitted as one independent Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_PROFILE=llmjp
source "${SCRIPT_DIR}/_eval_mdx_common.sh"

SAMPLE_NUM=16
SPLIT="test"
NUM_TEST_SAMPLE=-1
PROMPT_TYPE="llmjp4-harmony"
DATA_NAMES="math500,aime24,minerva_math,olympiadbench,amc23,aime26,gpqa_diamond,mmlu_stem,livecodebench_v6"
MAX_TOKENS_PER_CALL=38912
SEED=42
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=-1
START=0
END=-1
PIPELINE_PARALLEL_SIZE=1

# One experiment = one Slurm job.
GPUS_PER_EXP=8
CPUS_PER_EXP=64

# Same list style as sh/nict/eval_ray.sh: uncomment/add as many experiments as needed.
EXP_NAMES=(
  "exp002_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-LEASH4K"
  # "exp004_DAPO-8B-RPP-16K-N16-B128-LR1e-6-SYNC-1E-Overlong4K"
  # "sci-llm-jp-4-8b-sft"
  # "ELYZA-Thinking-1.0-llm-jp-4-32b-a3b"
  # "dapo-sci-sft-rl100k-enja-v3-e0a-tis-step360"
)

mdx_submit_experiments "${BASH_SOURCE[0]}"
