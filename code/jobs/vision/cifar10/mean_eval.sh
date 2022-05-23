#!/bin/bash

cd ../../../

for seed in 7 11 13
do
  parallel -j 8 \
    "python vision_mean_eval.py seed=${seed} target_weight_file={}vision_contrastive_model.pt dataset=vision/cifar10 gpu_id={#}" :::: jobs/vision/cifar10/seed_${seed}-contrastive_wights_path.txt
done

for seed in 7 11 13
do
  parallel -j 8 \
    "python vision_mean_eval.py seed=${seed} target_weight_file={}vision_contrastive_model.pt dataset=vision/cifar10 eval_all_checkpoints=false normalize=true gpu_id={#}" :::: jobs/vision/cifar10/seed_${seed}-contrastive_wights_path.txt
done
