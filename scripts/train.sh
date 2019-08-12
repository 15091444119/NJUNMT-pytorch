#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
ExpName="transformer_nist_zh2en_bpe"
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "./configs/transformer_nist_zh2en_bpe.yaml" \
    --log_path "./exp/$ExpName/log" \
    --saveto "./exp/$ExpName/save/" \
    --use_gpu