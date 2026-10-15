#!/bin/bash
#SBATCH -J meta_exp6_manifest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH:-}

PROJECT_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift"
MODEL_PATH="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/Kvasir/Other_LLM/Qwen3-VL-8B-Instruct"
TRAIN_DATA="${PROJECT_DIR}/data/medical_vqa_train_rl.json"
VAL_DATA="${PROJECT_DIR}/data/medical_vqa_val_rl.json"
PLUGIN_PATH="${PROJECT_DIR}/agentic_crop_plugin_async_medical_vqa.py"
SYSTEM_PROMPT="${PROJECT_DIR}/data/sys_prompt_gi_medical_vqa.txt"

RUN_NAME="medical-vqa-grpo-qwen3vl8b-colocate-local"
OUTPUT_DIR="${PROJECT_DIR}/output/${RUN_NAME}"

NUM_EPOCHS=3
LEARNING_RATE=1e-5
NUM_GENERATIONS=8
PER_DEVICE_TRAIN_BATCH=1
PER_DEVICE_EVAL_BATCH=1
GRAD_ACC_STEPS=1
MAX_COMPLETION_LENGTH=512
MAX_INPUT_LENGTH=7168
MAX_TURNS=3

VLLM_TP=2
VLLM_GPU_MEMORY_UTILIZATION=0.40
VLLM_MAX_MODEL_LEN=8192

EVAL_STEPS=200
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=5
LOGGING_STEPS=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NNODES=1
export NPROC_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export IMAGE_MAX_TOKEN_NUM=2048
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOG_REWARD_COMPONENTS=1
export REWARD_LOG_INTERVAL=1
export SWIFT_TIMEOUT=-1

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH:-}
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

for f in "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}" "${PLUGIN_PATH}" "${SYSTEM_PROMPT}"; do
    if [[ ! -e "${f}" ]]; then
        echo "[ERROR] Required path does not exist: ${f}"
        exit 1
    fi
done

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
GLOBAL_COMPLETION_BATCH=$((WORLD_SIZE * PER_DEVICE_TRAIN_BATCH * GRAD_ACC_STEPS))

if (( GLOBAL_COMPLETION_BATCH % NUM_GENERATIONS != 0 )); then
    echo "[ERROR] Invalid GRPO batch configuration"
    echo "WORLD_SIZE=${WORLD_SIZE}"
    echo "GLOBAL_COMPLETION_BATCH=${GLOBAL_COMPLETION_BATCH}"
    echo "NUM_GENERATIONS=${NUM_GENERATIONS}"
    exit 1
fi

ROLLOUT_PROMPTS=$((GLOBAL_COMPLETION_BATCH / NUM_GENERATIONS))

echo "========================================================================"
echo "[LOCAL MEDICAL VQA GRPO]"
echo "HOST                    = $(hostname)"
echo "CUDA_VISIBLE_DEVICES    = ${CUDA_VISIBLE_DEVICES}"
echo "WORLD_SIZE              = ${WORLD_SIZE}"
echo "GLOBAL_COMPLETION_BATCH = ${GLOBAL_COMPLETION_BATCH}"
echo "NUM_GENERATIONS         = ${NUM_GENERATIONS}"
echo "ROLLOUT_PROMPTS/ROUND   = ${ROLLOUT_PROMPTS}"
echo "VLLM_MODE               = colocate"
echo "VLLM_TP                 = ${VLLM_TP}"
echo "VLLM_GPU_MEM_UTIL       = ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "VLLM_MAX_MODEL_LEN      = ${VLLM_MAX_MODEL_LEN}"
echo "VLLM_ENABLE_LORA        = false"
echo "OUTPUT_DIR              = ${OUTPUT_DIR}"
echo "========================================================================"

swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --system "${SYSTEM_PROMPT}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs doc_crop_reward \
    --multi_turn_scheduler doc_crop_scheduler \
    --max_turns "${MAX_TURNS}" \
    --remove_unused_columns false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size "${VLLM_TP}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora false \
    --vllm_max_lora_rank 8 \
    --vllm_limit_mm_per_prompt '{"image": 4, "video": 0}' \
    --sleep_level 1 \
    --offload_model false \
    --offload_optimizer false \
    --num_generations "${NUM_GENERATIONS}" \
    --num_generations_eval "${NUM_GENERATIONS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --max_length "${MAX_INPUT_LENGTH}" \
    --temperature 1.0 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing true \
    --padding_free true \
    --deepspeed zero2 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --warmup_ratio 0.05 \
    --beta 0.04 \
    --epsilon 0.20 \
    --epsilon_high 0.28 \
    --importance_sampling_level sequence \
    --dynamic_sample true \
    --max_resample_times 1 \
    --dataset_shuffle true \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --dataloader_drop_last true \
    --load_from_cache_file false \
    --eval_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version false \
    --create_checkpoint_symlink true