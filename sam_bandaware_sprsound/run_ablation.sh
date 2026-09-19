#!/bin/bash
# Ablation Experiment Runner
# 5 experiments: Baseline, +A, +A+B, +A+B+C, Full

set -e

COMMON="--epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz"

echo "===== Experiment 1: Baseline ====="
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/baseline

echo "===== Experiment 2: +A (Same-band PatchMix) ====="
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/plusA

echo "===== Experiment 3: +A+B (Same-band + Band Importance) ====="
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/plusAB

echo "===== Experiment 4: +A+B+C (+ Freq-weighted λ) ====="
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusABC

echo "===== Experiment 5: Full (A+B+C+D) ====="
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --checkpoint_dir ./checkpoints/full

echo "===== All experiments done! ====="