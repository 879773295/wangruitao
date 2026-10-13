#!/bin/bash
# SPRSound 事件级 7 分类 — 四模块消融实验（6 组，与 sam_bandaware_sprsound 版设计一致）
# 基线：SAM + HF AST + MeanPool + Dropout
# 创新点：A=Same-band PatchMix, B=Band Importance, C=Freq-weighted λ, D=BandCL Loss
#
# 用法：
#   bash run_ablation.sh            # 完整 20 epoch × 6 组
#   EPOCHS=1 bash run_ablation.sh   # 每组 1 个 epoch
#   bash run_ablation.sh smoke      # 冒烟：Exp0+Exp5 各 5 个 batch，验证能跑通
#   START_FROM=3 bash run_ablation.sh   # 断点续跑：跳过 Exp0-2，从 Exp3 开始

set -e

PY=${PY:-python}
EPOCHS=${EPOCHS:-20}
START_FROM=${START_FROM:-0}
COMMON="--epochs $EPOCHS --batch_size 8 --lr 1e-5 --num_classes 7 --bandcl_weight 0.01 --data_path ./sprsound_event_7class.npz"

# 冒烟模式：只验证代码能跑通（1 epoch × 5 batch × 24 条测试样本），不出正式结果
if [ "${1:-}" = "smoke" ]; then
  SMOKE="--epochs 1 --max_steps 5 --test_limit 24"
  echo "===== SMOKE Exp0: 纯 SAM（无 PatchMix 无模块）====="
  HF_HUB_OFFLINE=1 $PY train_event.py $COMMON $SMOKE \
    --checkpoint_dir ./checkpoints/smoke_e0
  echo "===== SMOKE Exp5: Full（A+B+C+D 全开）====="
  HF_HUB_OFFLINE=1 $PY train_event.py $COMMON $SMOKE \
    --use_patch_mix --use_same_band_mix --use_band_importance \
    --use_freq_weighted_lam --use_bandcl \
    --checkpoint_dir ./checkpoints/smoke_full
  echo "===== SMOKE 通过：基线分支与 Full 分支均可运行 ====="
  exit 0
fi

if [ 0 -ge "$START_FROM" ]; then
echo "===== Exp 0: 纯 SAM (Baseline) ====="
echo "配置：SAM 优化器，无 PatchMix，无任何创新模块"
HF_HUB_OFFLINE=1 $PY train_event.py $COMMON \
  --checkpoint_dir ./checkpoints/abl_e0
fi

if [ 1 -ge "$START_FROM" ]; then
echo "===== Exp 1: SAM + PatchMix ====="
echo "配置：在纯 SAM 基础上添加原始 PatchMix（随机交换 patch）"
HF_HUB_OFFLINE=1 $PY train_event.py $COMMON \
  --use_patch_mix \
  --checkpoint_dir ./checkpoints/abl_patchmix
fi

if [ 2 -ge "$START_FROM" ]; then
echo "===== Exp 2: SAM + A ====="
echo "配置：用 A（Same-band PatchMix）替换原始 PatchMix"
HF_HUB_OFFLINE=1 $PY train_event.py $COMMON \
  --use_patch_mix --use_same_band_mix \
  --checkpoint_dir ./checkpoints/abl_plusA
fi

if [ 3 -ge "$START_FROM" ]; then
echo "===== Exp 3: SAM + A + B ====="
echo "配置：在 A 基础上添加 B（Band Importance Estimator）"
HF_HUB_OFFLINE=1 $PY train_event.py $COMMON \
  --use_patch_mix --use_same_band_mix --use_band_importance \
  --checkpoint_dir ./checkpoints/abl_plusAB
fi

if [ 4 -ge "$START_FROM" ]; then
echo "===== Exp 4: SAM + A + B + C ====="
echo "配置：在 A+B 基础上添加 C（Frequency-weighted λ）"
HF_HUB_OFFLINE=1 $PY train_event.py $COMMON \
  --use_patch_mix --use_same_band_mix --use_band_importance --use_freq_weighted_lam \
  --checkpoint_dir ./checkpoints/abl_plusABC
fi

if [ 5 -ge "$START_FROM" ]; then
echo "===== Exp 5: SAM + A + B + C + D (Full) ====="
echo "配置：在 A+B+C 基础上添加 D（BandCL Loss，权重 0.01）"
HF_HUB_OFFLINE=1 $PY train_event.py $COMMON \
  --use_patch_mix --use_same_band_mix --use_band_importance --use_freq_weighted_lam --use_bandcl \
  --checkpoint_dir ./checkpoints/abl_full
fi

echo "===== All 6 experiments done! ====="
echo "汇总各组 best Score 后填入 README 消融对比表"
