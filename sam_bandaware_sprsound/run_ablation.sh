#!/bin/bash
# Ablation Experiment Runner (6 experiments)
# 从原始 SAM 到完整 SAM Band-aware

set -e

COMMON="--epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz"

echo "===== Experiment 0: Original SAM (Baseline) ====="
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_patch_mix --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/original_sam

echo "===== Experiment 1: SAM + PatchMix ====="
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/patchmix

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