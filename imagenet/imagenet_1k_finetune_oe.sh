#!/bin/bash

python imagenet_1k_finetune_oe.py imagenet-1k \
    --model resnet50 --epochs 5 --momentum 0.9 --learning_rate 0.0001 \
    --decay 0.0001 --batch_size 64 --oe_batch_size 64 --score_type msp






