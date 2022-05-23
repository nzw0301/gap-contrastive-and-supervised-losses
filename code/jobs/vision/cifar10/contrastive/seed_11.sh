#!/bin/bash

cd ../../../../

seed=11

for n in 4 16 32 64 128 512
do
    for lr in 0.03125 0.0625 0.09375
    do
        pkill -9 python
        python -m torch.distributed.launch --nproc_per_node=8 --use_env vision_contrastive.py \
            loss.neg_size=${n} optimizer.lr=${lr} seed=${seed} contrastive=vision/cifar10 dataset=vision/cifar10
        pkill -9 python
    done
done
