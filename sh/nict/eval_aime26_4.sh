#!/usr/bin/env bash

set -uo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_local_config.sh"
export RAY_DATA_HOME="${RAY_DATA_HOME:?Set RAY_DATA_HOME in .env}"

SAMPLE_NUM=16
SPLIT="test"
NUM_TEST_SAMPLE=-1

# 分成两组，确保两个进程不会写同一个数据集目录
DATA_GROUP_0="math500,aime24,minerva_math"
DATA_GROUP_1="olympiadbench,amc23,aime26"

EXP_NAMES=(
  "DAPO-DSDT-1.5B-unsup-5e-5_20"
  "DAPO-DSDT-1.5B-unsup-5e-5_30"
  "DAPO-DSDT-1.5B-unsup-5e-5_40"
  "DAPO-DSDT-1.5B-unsup-5e-5_50"
  "DAPO-DSDT-1.5B-unsup-5e-5_60"
)

for exp_name in "${EXP_NAMES[@]}"; do
    PROMPT_TYPE="qwen25-math-cot"
    MODEL_NAME_OR_PATH="${RAY_DATA_HOME}/models/${exp_name}"
    OUTPUT_DIR="${exp_name}_${SAMPLE_NUM}"

    mkdir -p "${OUTPUT_DIR}"

    echo "======================================================"
    echo "▶ Starting evaluation for: ${exp_name}"
    echo "  - GPUs 0-3: ${DATA_GROUP_0}"
    echo "  - GPUs 4-7: ${DATA_GROUP_1}"
    echo "  - TP=4, PP=1"
    echo "  - Output Dir: ${OUTPUT_DIR}"
    echo "======================================================"

    # 第一组：GPU 0-3，TP=4
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    TOKENIZERS_PARALLELISM=false \
    "${EVAL_PYTHON}" -u math_eval.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_names "${DATA_GROUP_0}" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample "${NUM_TEST_SAMPLE}" \
        --max_tokens_per_call 38912 \
        --seed 42 \
        --temperature 0.6 \
        --n_sampling "${SAMPLE_NUM}" \
        --top_p 0.95 \
        --top_k 20 \
        --start 0 \
        --end -1 \
        --pipeline_parallel_size 1 \
        --use_vllm \
        --apply_chat_template \
        --save_outputs \
        > "${OUTPUT_DIR}/gpu_0_3.log" 2>&1 &

    PID0=$!

    # 第二组：GPU 4-7，TP=4
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    TOKENIZERS_PARALLELISM=false \
    "${EVAL_PYTHON}" -u math_eval.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_names "${DATA_GROUP_1}" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample "${NUM_TEST_SAMPLE}" \
        --max_tokens_per_call 38912 \
        --seed 42 \
        --temperature 0.6 \
        --n_sampling "${SAMPLE_NUM}" \
        --top_p 0.95 \
        --top_k 20 \
        --start 0 \
        --end -1 \
        --pipeline_parallel_size 1 \
        --use_vllm \
        --apply_chat_template \
        --save_outputs \
        > "${OUTPUT_DIR}/gpu_4_7.log" 2>&1 &

    PID1=$!

    echo "  - Worker 0 PID: ${PID0}"
    echo "  - Worker 1 PID: ${PID1}"

    STATUS0=0
    STATUS1=0

    wait "${PID0}" || STATUS0=$?
    wait "${PID1}" || STATUS1=$?

    if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
        echo "✗ Evaluation failed for: ${exp_name}"
        echo "  - GPU 0-3 exit code: ${STATUS0}"
        echo "  - GPU 4-7 exit code: ${STATUS1}"
        echo "  - Logs:"
        echo "      ${OUTPUT_DIR}/gpu_0_3.log"
        echo "      ${OUTPUT_DIR}/gpu_4_7.log"
        exit 1
    fi

    echo "✓ Finished evaluation for: ${exp_name}"
    echo
done
