#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=20:00:00

source "${HOME}/.bashrc"
cd "${PBS_O_WORKDIR:-.}"
[[ ! -f .env ]] || { set -a; source .env; set +a; }

source /etc/profile.d/modules.sh
# Use the same CUDA 13 build and uv project as the MDX evaluation.
module purge
module load "${NICT_CUDA_MODULE:-${EVAL_CUDA_MODULE:-${CUDA_MODULE_LLMJP:-cuda/13.0/13.0.3}}}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PWD}/.venv}"
[[ "${UV_PROJECT_ENVIRONMENT}" == /* ]] || UV_PROJECT_ENVIRONMENT="${PWD}/${UV_PROJECT_ENVIRONMENT}"
EVAL_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
[[ -x "${EVAL_PYTHON}" ]] || { echo "ERROR: missing unified uv Python: ${EVAL_PYTHON}" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PROMPT_TYPE="qwen25-math-cot"
SAMPLE_NUM=16

exp_name='Qwen3-4B-avgem-dapo-aime24'

RAY_DATA_HOME=${RAY_DATA_HOME:?Set RAY_DATA_HOME or source .env}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${RAY_DATA_HOME}/models/${exp_name}"}
# MODEL_NAME_OR_PATH="Rinn000/Qwen3-4B-em-dapo-aime"
OUTPUT_DIR="${exp_name}_${SAMPLE_NUM}"

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="math500,minerva_math,olympiadbench,aime24,amc23"
#DATA_NAME="aime24,amc23"
TOKENIZERS_PARALLELISM=false \
"${EVAL_PYTHON}" -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_names ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --max_tokens_per_call 4096 \
    --seed 42 \
    --temperature 0.6 \
    --n_sampling ${SAMPLE_NUM} \
    --top_p 1 \
    --top_k -1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs 

# # English multiple-choice datasets
# DATA_NAME="aqua,sat_math,mmlu_stem"
# TOKENIZERS_PARALLELISM=false \
# "${EVAL_PYTHON}" -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --num_shots 5

# # Chinese gaokao collections
# DATA_NAME="gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa"
# TOKENIZERS_PARALLELISM=false \
# "${EVAL_PYTHON}" -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot

# # Chinese other datasets
# DATA_NAME="cmath,cn_middle_school"
# TOKENIZERS_PARALLELISM=false \
# "${EVAL_PYTHON}" -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot


# # English competition datasets
# DATA_NAME="aime24"
# TOKENIZERS_PARALLELISM=false \
# "${EVAL_PYTHON}" -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs 
