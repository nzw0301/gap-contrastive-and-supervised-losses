#!/bin/bash

cd ../../../

for seed in 7 11 13
do
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=3029 loss.neg_size={1} optimizer.lr={2} gpu_id={#}" \
      ::: 8 64 256 1024 ::: 0.025 0.05
    pkill -9 python
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=3029 loss.neg_size={1} optimizer.lr={2} gpu_id={#}" \
      ::: 8 64 256 1024 ::: 0.075 0.1
    pkill -9 python
done

c=2000
epochs=135
for seed in 7 11 13
do
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=${c} loss.neg_size={1} optimizer.lr={2} gpu_id={#} epochs=${epochs}" \
      ::: 8 64 256 1024 ::: 0.025 0.05
    pkill -9 python
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=${c} loss.neg_size={1} optimizer.lr={2} gpu_id={#} epochs=${epochs}" \
      ::: 8 64 256 1024 ::: 0.075 0.1
    pkill -9 python
done

c=1000
epochs=270
for seed in 7 11 13
do
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=${c} loss.neg_size={1} optimizer.lr={2} gpu_id={#} epochs=${epochs}" \
      ::: 8 64 256 1024 ::: 0.025 0.05
    pkill -9 python
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=${c} loss.neg_size={1} optimizer.lr={2} gpu_id={#} epochs=${epochs}" \
      ::: 8 64 256 1024 ::: 0.075 0.1
    pkill -9 python
done

c=500
epochs=540
for seed in 7 11 13
do
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=${c} loss.neg_size={1} optimizer.lr={2} gpu_id={#} epochs=${epochs}" \
      ::: 8 64 256 1024 ::: 0.025 0.05
    pkill -9 python
    parallel -j 8 \
      "python language_contrastive.py seed=${seed} dataset.num_used_classes=${c} loss.neg_size={1} optimizer.lr={2} gpu_id={#} epochs=${epochs}" \
      ::: 8 64 256 1024 ::: 0.075 0.1
    pkill -9 python
done
