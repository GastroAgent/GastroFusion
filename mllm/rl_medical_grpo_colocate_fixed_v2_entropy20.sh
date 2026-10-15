#!/bin/bash

echo "Node: $(hostname)"
echo "NODE_RANK=${NODE_RANK:-0}"
echo "NNODES=${NNODES:-1}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE:-8}"


# =============================================================================
# USER CONFIG -- 以后主要改这里
# =============================================================================

PROJECT_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift"


# -----------------------------------------------------------------------------
# 1. 模型权重
# -----------------------------------------------------------------------------

MODEL_PATH="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/weights/Qwen3-VL-8B-Instruct-RL1-last/Qwen3-VL-8B-Instruct"


# -----------------------------------------------------------------------------
# 2. 训练集 / 验证集
# -----------------------------------------------------------------------------

TRAIN_DATA="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/data/pretrain/RL_2/filtered_train.json"

VAL_DATA="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/data/pretrain/RL_2/filtered_train.json"


# -----------------------------------------------------------------------------
# 3. Reward plugin / system prompt
# -----------------------------------------------------------------------------

PLUGIN_PATH="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/medical_vqa_reward.py"
SYSTEM_PROMPT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/data/sys_prompt_gi_medical_vqa.txt"


# -----------------------------------------------------------------------------
# 4. 输出
# -----------------------------------------------------------------------------

RUN_NAME="medical-vqa-grpo-qwen3vl8b-colocate-${SLURM_JOB_ID:-local}"

OUTPUT_DIR="${PROJECT_DIR}/output/${RUN_NAME}"


# -----------------------------------------------------------------------------
# 5. GRPO
# -----------------------------------------------------------------------------

NUM_EPOCHS=3

LEARNING_RATE=1e-5

# 训练：每个 prompt 固定 8 个 rollout
NUM_GENERATIONS=16

# 评估单独使用 2 个 rollout：
# WORLD_SIZE=2, PER_DEVICE_EVAL_BATCH=1 -> global eval batch=2，
# 因此 NUM_GENERATIONS_EVAL=2 可整除，同时比 eval G=8 更省显存/时间。
NUM_GENERATIONS_EVAL=2

PER_DEVICE_TRAIN_BATCH=1
PER_DEVICE_EVAL_BATCH=1

# GRPO 默认 generation batch =
# WORLD_SIZE * PER_DEVICE_TRAIN_BATCH * GRAD_ACC_STEPS
# 2 * 1 * 4 = 8，可被 NUM_GENERATIONS=8 整除。
GRAD_ACC_STEPS=4

MAX_COMPLETION_LENGTH=512

MAX_INPUT_LENGTH=7168

MAX_TURNS=1


# -----------------------------------------------------------------------------
# 6. A100-40G + colocate vLLM
# -----------------------------------------------------------------------------

VLLM_TP=2

VLLM_GPU_MEMORY_UTILIZATION=0.40

VLLM_MAX_MODEL_LEN=8192


# -----------------------------------------------------------------------------
# 7. eval / save
# -----------------------------------------------------------------------------

EVAL_STEPS=200

SAVE_STEPS=200

SAVE_TOTAL_LIMIT=5

LOGGING_STEPS=1


# =============================================================================
# ENV
# =============================================================================

mkdir -p "${OUTPUT_DIR}"

export IMAGE_MAX_TOKEN_NUM=2048

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH:-}


# -----------------------------------------------------------------------------
# Reward 组件日志
#
# 这里仍然保留。
# 你的 reward plugin 内部即使没有 wandb，也不会因此报错。
# -----------------------------------------------------------------------------

export LOG_REWARD_COMPONENTS=1

export REWARD_LOG_INTERVAL=1


# -----------------------------------------------------------------------------
# Swift timeout
# -----------------------------------------------------------------------------

export SWIFT_TIMEOUT=-1


# =============================================================================
# SANITY CHECK
# =============================================================================

for f in \
    "${MODEL_PATH}" \
    "${TRAIN_DATA}" \
    "${VAL_DATA}" \
    "${PLUGIN_PATH}" \
    "${SYSTEM_PROMPT}"
do

    if [[ ! -e "${f}" ]]; then

        echo "[ERROR] Required path does not exist:"
        echo "        ${f}"

        exit 1

    fi

done


# =============================================================================
# GRPO BATCH CHECK
# =============================================================================

WORLD_SIZE=$(( ${NNODES:-1} * ${NPROC_PER_NODE:-8} ))

GLOBAL_COMPLETION_BATCH=$((
    WORLD_SIZE
    * PER_DEVICE_TRAIN_BATCH
    * GRAD_ACC_STEPS
))

GLOBAL_EVAL_BATCH=$((
    WORLD_SIZE
    * PER_DEVICE_EVAL_BATCH
))


if (( GLOBAL_COMPLETION_BATCH % NUM_GENERATIONS != 0 )); then

    echo "[ERROR] Invalid GRPO TRAIN batch configuration"

    echo "WORLD_SIZE              = ${WORLD_SIZE}"
    echo "PER_DEVICE_TRAIN_BATCH  = ${PER_DEVICE_TRAIN_BATCH}"
    echo "GRAD_ACC_STEPS          = ${GRAD_ACC_STEPS}"
    echo "GLOBAL_COMPLETION_BATCH = ${GLOBAL_COMPLETION_BATCH}"
    echo "NUM_GENERATIONS         = ${NUM_GENERATIONS}"

    exit 1

fi


if (( GLOBAL_EVAL_BATCH % NUM_GENERATIONS_EVAL != 0 )); then

    echo "[ERROR] Invalid GRPO EVAL batch configuration"

    echo "WORLD_SIZE              = ${WORLD_SIZE}"
    echo "PER_DEVICE_EVAL_BATCH   = ${PER_DEVICE_EVAL_BATCH}"
    echo "GLOBAL_EVAL_BATCH       = ${GLOBAL_EVAL_BATCH}"
    echo "NUM_GENERATIONS_EVAL    = ${NUM_GENERATIONS_EVAL}"

    exit 1

fi


ROLLOUT_PROMPTS=$((
    GLOBAL_COMPLETION_BATCH
    / NUM_GENERATIONS
))


# =============================================================================
# PRINT CONFIG
# =============================================================================

echo "========================================================================"
echo "[MEDICAL VQA GRPO CONFIG]"
echo
echo "MODEL_PATH               = ${MODEL_PATH}"
echo "TRAIN_DATA               = ${TRAIN_DATA}"
echo "VAL_DATA                 = ${VAL_DATA}"
echo "PLUGIN_PATH              = ${PLUGIN_PATH}"
echo "SYSTEM_PROMPT            = ${SYSTEM_PROMPT}"
echo "OUTPUT_DIR               = ${OUTPUT_DIR}"
echo
echo "WORLD_SIZE               = ${WORLD_SIZE}"
echo "GLOBAL_COMPLETION_BATCH  = ${GLOBAL_COMPLETION_BATCH}"
echo "NUM_GENERATIONS(train)   = ${NUM_GENERATIONS}"
echo "ROLLOUT_PROMPTS/ROUND    = ${ROLLOUT_PROMPTS}"
echo "GLOBAL_EVAL_BATCH        = ${GLOBAL_EVAL_BATCH}"
echo "NUM_GENERATIONS(eval)    = ${NUM_GENERATIONS_EVAL}"
echo
echo "VLLM_MODE                = colocate"
echo "VLLM_TP                  = ${VLLM_TP}"
echo "VLLM_GPU_MEM_UTIL        = ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "VLLM_MAX_MODEL_LEN       = ${VLLM_MAX_MODEL_LEN}"
echo "VLLM_ENABLE_LORA         = false  # workaround for Qwen3-VL/vLLM multimodal LoRA init"
echo
echo "REPORT_TO                = none"
echo "========================================================================"


# =============================================================================
# TRAIN
# =============================================================================

swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --system "${SYSTEM_PROMPT}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs medical_vqa_reward \
    --remove_unused_columns false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size "${VLLM_TP}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora false \
    --vllm_max_lora_rank 64 \
    --vllm_limit_mm_per_prompt '{"image": 4, "video": 0}' \
    --sleep_level 1 \
    --offload_model false \
    --offload_optimizer false \
    --num_generations "${NUM_GENERATIONS}" \
    --num_generations_eval "${NUM_GENERATIONS_EVAL}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --max_length "${MAX_INPUT_LENGTH}" \
    --temperature 1.0 \
    --tuner_type lora \
    --lora_rank 64 \
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
    --log_entropy true \
    --top_entropy_quantile 0.20 \
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