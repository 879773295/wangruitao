#!/bin/bash
# 消融实验脚本（6 组实验）
# 基础模型：SAM（Sharpness-Aware Minimization）
# 创新点：A=Same-band PatchMix, B=Band Importance, C=Freq-weighted λ, D=BandCL Loss

set -e

COMMON="--epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 --bandcl_weight 0.01 --data_path sprsound_data.npz"

echo "===== Exp 0: 纯 SAM (Baseline) ====="
echo "配置：SAM 优化器，无 PatchMix，无任何创新模块"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --no_patch_mix \
  --no_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/pure_sam

echo "===== Exp 1: SAM + PatchMix ====="
echo "配置：在纯 SAM 基础上添加原始 PatchMix（随机交换 patch）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --no_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/patchmix

echo "===== Exp 2: SAM + A ====="
echo "配置：用 A（Same-band PatchMix）替换原始 PatchMix"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusA

echo "===== Exp 3: SAM + A + B ====="
echo "配置：在 A 基础上添加 B（Band Importance Estimator）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusAB

echo "===== Exp 4: SAM + A + B + C ====="
echo "配置：在 A+B 基础上添加 C（Frequency-weighted λ）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --use_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusABC

echo "===== Exp 5: SAM + A + B + C + D (Full) ====="
echo "配置：在 A+B+C 基础上添加 D（BandCL Loss）"
HF_HUB_OFFLINE=1 python train.py $COMMON \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --use_freq_weighted_lam \
  --use_bandcl \
  --checkpoint_dir ./checkpoints/full

echo "===== All experiments done! ====="
