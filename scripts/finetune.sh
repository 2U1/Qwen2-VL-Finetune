#!/bin/bash

# You can use 2B instead of 7B
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH
# export NCCL_DEBUG="INFO"
deepspeed src/training/train.py \
    --model_id $MODEL_NAME \
    --data_path /home/mk.thomas/llmops/data/ml/qwen-7b-VL/v1/data/result/combined_test.json \
    --image_folder /home/mk.thomas/llmops/data/ml/qwen-7b-VL/v1/data/images \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/fft_0912 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --min_pixels $((512 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
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
    --dataloader_num_workers 1