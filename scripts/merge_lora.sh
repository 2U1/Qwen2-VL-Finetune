#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /path/to/model \
    --model-base $MODEL_NAME  \
    --save-model-path /path/to/save \
    --safe-serialization