#!/bin/bash

cd ../../../

for seed in 7 11 13
do
  for lr in 0.05 0.01 0.005 0.001 0.0005
  do
    parallel -j 8 \
      "python language_linear_eval.py seed=${seed} optimizer.lr=${lr} target_weight_file={}language_contrastive_model.pt gpu_id={#}" :::: jobs/language/seed_${seed}-contrastive_wights_path.txt
    pkill -9 python
  done
done
