#!/usr/bin/env bash
# MDX/Slurm equivalent of sh/nict/eval_ray.sh.
# Edit EXP_NAMES below; every enabled entry is submitted as one independent Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
export EVAL_PROFILE=dsdt
source "${SCRIPT_DIR}/_eval_mdx_common.sh"

SAMPLE_NUM=16
SPLIT="test"
NUM_TEST_SAMPLE=-1
PROMPT_TYPE="qwen25-math-cot"
DATA_NAMES="math500,aime24,minerva_math,olympiadbench,amc23,aime26,gpqa_diamond,mmlu_stem,livecodebench_v6"
MAX_TOKENS_PER_CALL=38912
SEED=42
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
START=0
END=-1
PIPELINE_PARALLEL_SIZE=1

# One experiment = one Slurm job.
GPUS_PER_EXP=4
CPUS_PER_EXP=64

# Same list style as sh/nict/eval_ray.sh: uncomment/add as many experiments as needed.
EXP_NAMES=(
  # "DAPO-DSDT-1.5B-RELAXED-LASER-L2048-a0.5"
  # "DAPO-DSDT-1.5B-RELAXED-LASER-L2048-a1.25"
  "DAPO-DSDT-1.5B-LASER-L1024-a1.25"
  # "DAPO-DSDT-1.5B-LASER-L2048-a1.25"
  "DAPO-DSDT-1.5B-LASER-L3072-a1.25"
  "DAPO-DSDT-1.5B-LASER-L4096-a1.25"
  # "DAPO-DSDT-1.5B-LP-C-1.5e-4"
  # "DAPO-DSDT-1.5B-LP-C-1e-3"
  # "DAPO-DSDT-1.5B-LP-C-3e-4"
  # "DAPO-DSDT-1.5B-LP-C-5e-4"
  # "DAPO-DSDT-1.5B-LP-C-5e-5"
  # "DAPO-DSDT-1.5B-LP-C-1e-4"
  # "DAPO-DSDT-1.5B-LP-D-5e-4-m0.5"
  # "DAPO-DSDT-1.5B-LP-D-5e-5-m0.5"
  # "DAPO-DSDT-1.5B-LP-D-1e-4-m0.5"
  # "DAPO-DSDT-1.5B-LP-D-3e-4-m0.5"
  # "DAPO-DSDT-1.5B-LP-D-1e-3-m0.5"
  # "DAPO-DSDT-1.5B-LP-D-1.5e-4-m0.5"
  # "DAPO-DSDT-1.5B-0-1024"
  # "DAPO-DSDT-1.5B-0-1536"
  # "DAPO-DSDT-1.5B-0-2048"
  # "DAPO-DSDT-1.5B-0-3072"
  # "DAPO-DSDT-1.5B-smtm-1.5e-4"
  # "DAPO-DSDT-1.5B-smtm-3e-4"
  # "DAPO-DSDT-1.5B-smtm-5e-5"
  # "DAPO-DSDT-1.5B-LASER-D-LB256-a0.5"
  # "DAPO-DSDT-1.5B-LASER-D-LB512-a0.5"
  # "DAPO-DSDT-1.5B-LASER-D-LB1024-a0.5"
  # "DAPO-DSDT-1.5B-LASER-D-LB2048-a0.5"
  # "DAPO-DSDT-1.5B-LASER-D-LB3072-a0.5"
  # "DAPO-DSDT-1.5B-LASER-D-LB4096-a0.5"
  # "DAPO-DSDT-1.5B-LASER-L1024-a0.5"
  # "DAPO-DSDT-1.5B-LASER-L2048-a0.5"
  # "DAPO-DSDT-1.5B-LASER-L3072-a0.5"
  # "DAPO-DSDT-1.5B-LASER-L4096-a0.5"
  # "DAPO-DSDT-1.5B-1.5e-4"
  # "DAPO-DSDT-1.5B-1e-3"
  # "DAPO-DSDT-1.5B-1e-4"
  # "DAPO-DSDT-1.5B-1e-5"
  # "DAPO-DSDT-1.5B-2e-5"
  # "DAPO-DSDT-1.5B-3e-4"
  # "DAPO-DSDT-1.5B-3e-5"
  # "DAPO-DSDT-1.5B-4e-5"
  # "DAPO-DSDT-1.5B-5e-3"
  # "DAPO-DSDT-1.5B-5e-4"
  # "DAPO-DSDT-1.5B-5e-5"
  # "DAPO-DSDT-1.5B-6e-5"
  # "DAPO-DSDT-1.5B-7e-5"
  # "DAPO-DSDT-1.5B-8e-5"
  # "DAPO-DSDT-1.5B-9e-5"
  # "DAPO-DSDT-1.5B-LEASH-Lt1024-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt128-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt1536-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt2048-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt256-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt4096-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt512-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt6144-llr0.005"
  # "DAPO-DSDT-1.5B-unsup-5e-5_10"
  # "DAPO-DSDT-1.5B-unsup-5e-5_15"
  # "DAPO-DSDT-1.5B-unsup-5e-5_20"
  # "DAPO-DSDT-1.5B-unsup-5e-5_25"
  # "DAPO-DSDT-1.5B-unsup-5e-5_30"
  # "DAPO-DSDT-1.5B-unsup-5e-5_35"
  # "DAPO-DSDT-1.5B-unsup-5e-5_40"
  # "DAPO-DSDT-1.5B-unsup-5e-5_45"
  # "DAPO-DSDT-1.5B-unsup-5e-5_5"
  # "DAPO-DSDT-1.5B-unsup-5e-5_50"
  # "DAPO-DSDT-1.5B-unsup-5e-5_55"
  # "DAPO-DSDT-1.5B-unsup-5e-5_60"
  # "DAPO-DSDT-1.5B-unsup-5e-5_65"
  # "DAPO-DSDT-1.5B-unsup-5e-5_70"
  # "DAPO-DSDT-1.5B-unsup-5e-5_75"
  # "DAPO-DSDT-1.5B-unsup-5e-5_80"
  # "DAPO-DSDT-1.5B-unsup-5e-5_85"
  # "DAPO-DSDT-1.5B-unsup-5e-5_90"
  # "DAPO-DSDT-7B-unsup-5e-5_10
  # "DAPO-DSDT-7B-unsup-5e-5_20
  # "DAPO-DSDT-7B-unsup-5e-5_30
  # "DAPO-DSDT-7B-unsup-5e-5_40
  # "DAPO-DSDT-7B-unsup-5e-5_45"
  # "DAPO-DSDT-7B-unsup-5e-5_50"
  # "DAPO-DSDT-7B-unsup-5e-5_55"
  # "DAPO-DSDT-7B-unsup-5e-5_60"
  # "DAPO-DSDT-7B-unsup-5e-5_65"
  # "DAPO-DSDT-7B-unsup-5e-5_70"
  # "DAPO-DSDT-7B-unsup-5e-5_75"
  # "DAPO-DSDT-7B-1.5e-4"
  # "DAPO-DSDT-7B-1e-3"
  # "DAPO-DSDT-7B-1e-4"
  # "DAPO-DSDT-7B-1e-5"
  # "DAPO-DSDT-7B-5e-3"
  # "DAPO-DSDT-7B-5e-4"
  # "DAPO-DSDT-7B-3e-4"
  # "DAPO-DSDT-7B-5e-5"
  # "DAPO-DSDT-7B-3e-5"
  # "DeepSeek-R1-Distill-Qwen-1.5B"
  # "DeepSeek-R1-Distill-Qwen-7B"
  # "DAPO-DSDT-1.5B-unsup-5e-5_10"
  # "DAPO-DSDT-1.5B-unsup-5e-5_45"
  # "DAPO-DSDT-1.5B-unsup-5e-5_55"
  # "DAPO-DSDT-1.5B-unsup-5e-5_20"
  # "DAPO-DSDT-1.5B-unsup-5e-5_30"
  # "DAPO-DSDT-1.5B-unsup-5e-5_40"
  # "DAPO-DSDT-1.5B-unsup-5e-5_50"
  # "DAPO-DSDT-1.5B-unsup-5e-5_60"
  # "DAPO-DSDT-1.5B-unsup-5e-5_65"
  # "DAPO-DSDT-1.5B-unsup-5e-5_70"
  # "DAPO-DSDT-1.5B-unsup-5e-5_75"
  # "DAPO-DSDT-1.5B-unsup-5e-5_80"
  # "DAPO-DSDT-1.5B-unsup-5e-5_85"
  # "DAPO-DSDT-1.5B-unsup-5e-5_90"
  # "DAPO-DSDT-1.5B-LEASH-Lt4096-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt6144-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt2048-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt1536-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt1024-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt512-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt256-llr0.005"
  # "DAPO-DSDT-1.5B-LEASH-Lt128-llr0.005"
)

mdx_submit_experiments "${BASH_SOURCE[0]}"
