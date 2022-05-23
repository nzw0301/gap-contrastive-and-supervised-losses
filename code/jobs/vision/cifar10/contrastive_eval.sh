#!/bin/bash

cd ../../../

for seed in 7 11 13
do
  parallel -j 8 \
    "python vision_contrastive_eval.py seed=${seed} dataset=vision/cifar10 target_weight_file={}vision_contrastive_model.pt gpu_id={#}" :::: jobs/vision/cifar10/seed_${seed}-contrastive_wights_path.txt
done
