#!/bin/bash

cd ../../../

for seed in 7 11 13
do
    parallel -j 8 \
      "python language_contrastive_eval.py seed=${seed} target_weight_file={}language_contrastive_model.pt gpu_id={#}" :::: jobs/language/seed_${seed}-contrastive_wights_path.txt
    pkill -9 python
done
