#!/bin/bash
export RAY_DATA_HOME=${RAY_DATA_HOME:-"/mnt/fern/zelin/verl"}

SAMPLE_NUM=16
SPLIT="test"
NUM_TEST_SAMPLE=-1
DATA_NAME="aime26,math500,minerva_math,olympiadbench,aime24,amc23"

EXP_NAMES=(
  "Qwen2.5-3B"
)
# 遍历每一个模型进行测试
for exp_name in "${EXP_NAMES[@]}"; do
    
    # 动态匹配 prompt_type (转换为小写进行匹配，防止大小写导致匹配失败)
    exp_lower="${exp_name,,}"
    if [[ "$exp_lower" == *"qwen"* ]]; then
        PROMPT_TYPE="qwen25-math-cot"
    elif [[ "$exp_lower" == *"llama"* ]]; then
        PROMPT_TYPE="deepseek-distill-llama-math-cot"
    else
        echo "[Warning] Unknown prompt type for $exp_name! Skipping..."
        continue
    fi

    # 动态构建路径和输出目录
    MODEL_NAME_OR_PATH="${RAY_DATA_HOME}/models/${exp_name}"
    OUTPUT_DIR="${exp_name}_${SAMPLE_NUM}"

    echo "======================================================"
    echo "▶ Starting evaluation for: ${exp_name}"
    echo "  - Prompt Type: ${PROMPT_TYPE}"
    echo "  - Output Dir: ${OUTPUT_DIR}"
    echo "======================================================"

    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_names ${DATA_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --max_tokens_per_call 4096\
        --seed 42 \
        --temperature 0.6 \
        --n_sampling ${SAMPLE_NUM} \
        --top_p 0.95 \
        --top_k 20 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs 

    echo "✓ Finished evaluation for: ${exp_name}"
    echo ""
done
