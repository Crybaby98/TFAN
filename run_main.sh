#!/bin/bash
/root/miniconda3/bin/python main.py \
    --dataset 'AWF900' \
    --dataset_dir './datasets/AWF900'\
    --way 100 \
    --shot 1 \
    --query 15 \
    --no_val \
    --val_epoch 5 \
    --val_trial 1000 \
    --time_gap 0 \
    --gpu 0 \
    --seed 42 \
    --lr 1e-2 \
    --weight_decay 2e-3 \
    --gamma 0.1 \
    --stage 4 \
    --stage_size 15
