#!/bin/bash

# You can use 2B instead of 7B
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/lora_vision_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --min_pixels $((256 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4