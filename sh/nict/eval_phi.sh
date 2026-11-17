#!/usr/bin/env bash
# MDX/Slurm equivalent of sh/nict/eval_ray_phi.sh.
# Edit EXP_NAMES below; every enabled entry is submitted as one independent Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_mdx_common.sh"

SAMPLE_NUM=16
SPLIT="test"
NUM_TEST_SAMPLE=-1
PROMPT_TYPE="phi4-mini-reasoning"
# Transfer evaluation datasets (optional). Keep obligation_transfer disabled until generated.
# DATA_NAMES="gsm_plus,gsm_plus_seed,gsm_symbolic_main,gsm_symbolic_p1,gsm_symbolic_p2,math_perturb_simple,math_perturb_hard,omega_exploratory,omega_compositional,omega_transformative"
# DATA_NAMES="processbench_gsm8k,proofbench"
# DATA_NAMES="obligation_transfer"
DATA_NAMES="gpqa_diamond,mmlu_stem,livecodebench_v6"
MAX_TOKENS_PER_CALL=38912
SEED=42
TEMPERATURE=0.8
TOP_P=0.95
TOP_K=20
START=0
END=-1
PIPELINE_PARALLEL_SIZE=1

# Same resources as the original Phi Ray submitter.
GPUS_PER_EXP=8
CPUS_PER_EXP=16

# Same list style as sh/nict/eval_ray_phi.sh.
EXP_NAMES=(
  # "Phi-4-mini-reasoning"
  # "Phi-4B-unsup-5e-5_10"
  # "Phi-4B-unsup-5e-5_20"
  # "Phi-4B-unsup-5e-5_30"
  # "Phi-4B-unsup-5e-5_40"
  # "Phi-4B-unsup-5e-5_50"
  # "Phi-4B-unsup-5e-5_60"
  # "Phi-4B-unsup-5e-5_70"
  # "Phi-4B-unsup-5e-5_80"
  # "Phi-4B-unsup-5e-5_90"
  # "Phi-4B-unsup-5e-5_100"
  # "Phi-4B-unsup-5e-5_110"
  # "Phi-4B-unsup-5e-5_120"
  "Phi-4B-5e-5"
  "Phi-4B-5e-4"
  "Phi-4B-1e-4"
  "Phi-4B-1e-3"
  "Phi-4B-3e-4"
  "Phi-4B-7e-4"
  # "Phi-4B-1e-2"
  # "Phi-4B-5e-3"
  # "Phi-4B-3e-3"
)

mdx_submit_experiments "${BASH_SOURCE[0]}"
