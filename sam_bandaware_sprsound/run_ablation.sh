#!/bin/bash
# Ablation Experiment Runner (6 experiments)
# 从原始 SAM 到完整 SAM Band-aware
# 基础模型：SAM 原版（Sharpness-Aware Minimization）

set -e

COMMON="--epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 --bandcl_weight 0.01 --data_path sprsound_data.npz"

echo "===== Experiment 0: 纯 SAM (Baseline) ====="
echo "配置：SAM 优化器，无 PatchMix，无任何创新模块"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_patch_mix \
  --no_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/pure_sam

echo "===== Experiment 1: SAM + 原始 PatchMix ====="
echo "配置：SAM + 随机交换 patch（原始 PatchMix）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --no_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/patchmix

echo "===== Experiment 2: +A (Same-band PatchMix) ====="
echo "配置：SAM + 同频段内交换 patch（A）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusA

echo "===== Experiment 3: +A+B (Same-band + Band Importance) ====="
echo "配置：A + 学习频段重要性权重（B）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusAB

echo "===== Experiment 4: +A+B+C (+ Freq-weighted λ) ====="
echo "配置：B + 用频段重要性加权混合比例（C）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --use_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusABC

echo "===== Experiment 5: Full (A+B+C+D) ====="
echo "配置：C + BandCL Loss 约束混合后频段特征一致性（D）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --use_freq_weighted_lam \
  --use_bandcl \
  --checkpoint_dir ./checkpoints/full

echo "===== All experiments done! ====="
