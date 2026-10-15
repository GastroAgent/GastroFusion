#!/bin/bash

# ============================================================
# Local single-node / 6-GPU SFT
# GPUs: 1,2,3,4,5,6
# ============================================================

cd /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH:-}

# Reduce CUDA memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Single-node distributed settings
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "============================================================"
echo "[LOCAL 6-GPU SFT]"
echo "HOSTNAME             = $(hostname)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE       = ${NPROC_PER_NODE}"
echo "NNODES               = ${NNODES}"
echo "NODE_RANK             = ${NODE_RANK}"
echo "MASTER_ADDR           = ${MASTER_ADDR}"
echo "MASTER_PORT           = ${MASTER_PORT}"
echo "============================================================"

swift sft \
    --model /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/weights/Qwen3-VL-8B-Instruct-917/Qwen3-VL-8B-Instruct \
    --tuner_type lora \
    --dataset /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/data/pretrain/Multi_Image/train_sft_f_18_clean1.json \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --group_by_length true \
    --output_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/output_918/Qwen3-VL-8B-Instruct-3epoch-sft-local-7gpu \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --system /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/data/sys_prompt_gi_medical_vqa.txt \
    --deepspeed zero3_offload \
    --attn_impl flash_attn \
    --padding_free true \
    --gradient_checkpointing true \
    --use_liger_kernel true
