#!/bin/bash
# 消融实验脚本
# 测试每个创新点的效果

PYTHON="C:/Users/diaoqingguang/miniconda3/envs/patchmix_torch201/python.exe"
export HF_HUB_OFFLINE=1

echo "=== 消融实验 ==="

# 配置 1：基线（SAM only）
echo "=== 配置 1：基线（SAM only）==="
$PYTHON -u train.py --epochs 20 --batch_size 8 --lr 1e-5 \
  --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/ablation_baseline \
  > ablation_baseline.txt 2>&1

# 配置 2：+ Same-band Patch-Mix
echo "=== 配置 2：+ Same-band Patch-Mix ==="
$PYTHON -u train.py --epochs 20 --batch_size 8 --lr 1e-5 \
  --use_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/ablation_sameband \
  > ablation_sameband.txt 2>&1

# 配置 3：+ Band Importance Estimator
echo "=== 配置 3：+ Band Importance Estimator ==="
$PYTHON -u train.py --epochs 20 --batch_size 8 --lr 1e-5 \
  --use_same_band_mix --use_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/ablation_bandimport \
  > ablation_bandimport.txt 2>&1

# 配置 4：+ Frequency-weighted λ
echo "=== 配置 4：+ Frequency-weighted λ ==="
$PYTHON -u train.py --epochs 20 --batch_size 8 --lr 1e-5 \
  --use_same_band_mix --use_band_importance --use_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/ablation_freqweight \
  > ablation_freqweight.txt 2>&1

# 配置 5：全部模块
echo "=== 配置 5：全部模块 ==="
$PYTHON -u train.py --epochs 20 --batch_size 8 --lr 1e-5 \
  --use_same_band_mix --use_band_importance --use_freq_weighted_lam --use_bandcl \
  --checkpoint_dir ./checkpoints/ablation_full \
  > ablation_full.txt 2>&1

echo "=== 消融实验完成 ==="
