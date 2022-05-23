#!/bin/bash

cd ../../..

for seed in 7 11 13
do
  parallel -j 8 \
    "python vision_mean_eval.py seed=${seed} target_weight_file={}vision_contrastive_model.pt gpu_id={#}" :::: jobs/vision/cifar100/seed_${seed}-contrastive_wights_path.txt
done

for seed in 7 11 13
do
  parallel -j 8 \
    "python vision_mean_eval.py seed=${seed} target_weight_file={}vision_contrastive_model.pt eval_all_checkpoints=false normalize=true gpu_id={#}" :::: jobs/vision/cifar100/seed_${seed}-contrastive_wights_path.txt
done
