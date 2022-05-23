#!/bin/bash

cd ../../..

for seed in 7 11 13
do
  cat jobs/vision/cifar100/seed_${seed}-contrastive_wights_path.txt | while read weight_dir; do
    pkill -9 python
    python -m torch.distributed.launch --nproc_per_node=8 --use_env vision_linear_eval.py \
      seed=${seed} target_weight_file=${weight_dir}vision_contrastive_model.pt
    pkill -9 python
  done
done
