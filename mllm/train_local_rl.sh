#!/bin/bash

# ============================================================
# Local Medical VQA GRPO Training
# ============================================================

set -e


# ============================================================
# Environment
# ============================================================

source /mnt/inaisfs/data/apps/cuda12.4.sh

source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop


export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH}

export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH}


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export IMAGE_MAX_TOKEN_NUM=2048

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LOG_REWARD_COMPONENTS=1

export REWARD_LOG_INTERVAL=1

export SWIFT_TIMEOUT=-1



# ============================================================
# Path
# ============================================================

PROJECT_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift"

cd ${PROJECT_DIR}


MODEL_PATH="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/weights/Qwen3-VL-8B-Instruct-918-v1-10200/Qwen3-VL-8B-Instruct"


TRAIN_DATA="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/data/pretrain/RL_1/train.json"


PLUGIN_PATH="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/medical_vqa_reward.py"


SYSTEM_PROMPT="${PROJECT_DIR}/data/sys_prompt_gi_medical_vqa.txt"


OUTPUT_DIR="${PROJECT_DIR}/output/medical-vqa-grpo-local"



mkdir -p ${OUTPUT_DIR}



# ============================================================
# Start
# ============================================================

swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL_PATH} \
    --dataset ${TRAIN_DATA} \
    --system ${SYSTEM_PROMPT} \
    --external_plugins ${PLUGIN_PATH} \
    --reward_funcs medical_vqa_reward \
    --remove_unused_columns false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.40 \
    --vllm_max_model_len 8192 \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora false \
    --vllm_limit_mm_per_prompt '{"image":4,"video":0}' \
    --num_generations 16 \
    --num_generations_eval 2 \
    --max_completion_length 512 \
    --max_length 7168 \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
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
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --report_to none \
    --run_name medical-vqa-grpo-local \
    --output_dir ${OUTPUT_DIR} \
    --add_version false \
    --create_checkpoint_symlink true