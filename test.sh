#!/usr/bin/env bash
set -xeuo pipefail

PROMPT_TYPE="qwen3-math-cot"

project_name='TEMPO'
exp_name='TEMPO-Qwen3-4B-thr-0.672-kl-0.05'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_DIR=${MODEL_DIR:-"${RAY_DATA_HOME}/model/${project_name}/${exp_name}"}
source "$(dirname -- "${BASH_SOURCE[0]}")/sh/nict/_local_config.sh"
accelerate launch --config_file ${ACCELERATE_CONFIG_FILE:?Set ACCELERATE_CONFIG_FILE in .env} --num_processes 8 lm_eval \
    --model hf \
    --model_args pretrained=${CKPTS_DIR},trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 0 \
    --batch_size 8 \
    --gen_kwargs '{"max_new_tokens": 4096, "do_sample": true, "temperature": 0.6, "top_p": 0.7, "top_k": 20}' \
    --log_samples \
    --output_path ./eval_results.json
#export CUDA_VISIBLE_DEVICES=0
MODEL_NAME_OR_PATH=${CKPTS_DIR}
OUTPUT_DIR="Qwen2.5-1.5B-EMPO-NM-COT-20K"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR
