#!/bin/bash
# ============================================================
#  在 RTX PRO 6000 上跑的 "超 62" 一键包（含 SupCon）
#  前提：项目代码已在本地（含 --max_epochs 补丁），python 指向 PRO 6000 环境
#  用法：bash pro6000_run.sh
# ============================================================
# 改成你的 PRO 6000 机器上的 python（base env 就行，torch 2.12）
PY="${PYTHON:-python}"
cd "$(dirname "$0")" || exit 1

SEEDS=(1 2 3 4 5 6 1337 42)
mkdir -p save/pro6000

echo "========== [1/2] 快训 8 个 seed (50ep LR 曲线, epoch 10 停, SupCon) =========="
for s in "${SEEDS[@]}"; do
  echo "--- seed${s} started $(date +%H:%M:%S) ---"
  $PY -u main.py --method patchmix_cl --use_supcon --epochs 50 --max_epochs 10 --batch_size 8 --num_workers 0 \
    --seed "$s" --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
    --from_sl_official --audioset_pretrained --ma_update --ma_beta 0.5 \
    --save_dir ./save/pro6000 --tag seed${s}_supcon > pro6000_seed${s}_supcon.txt 2>&1
  echo "--- seed${s} done $(date +%H:%M:%S) ---"
done

echo "========== [2/2] 8 成员集成 + TTA(16) + 阈值校准 =========="
$PY -u ensemble_eval.py \
  --ckpts ./save/pro6000/icbhi_ast_patchmix_cl_seed1_supcon/best.pth \
          ./save/pro6000/icbhi_ast_patchmix_cl_seed2_supcon/best.pth \
          ./save/pro6000/icbhi_ast_patchmix_cl_seed3_supcon/best.pth \
          ./save/pro6000/icbhi_ast_patchmix_cl_seed4_supcon/best.pth \
          ./save/pro6000/icbhi_ast_patchmix_cl_seed5_supcon/best.pth \
          ./save/pro6000/icbhi_ast_patchmix_cl_seed6_supcon/best.pth \
          ./save/pro6000/icbhi_ast_patchmix_cl_seed1337_supcon/best.pth \
          ./save/pro6000/icbhi_ast_patchmix_cl_seed42_supcon/best.pth \
  --tta 16 2>&1 | tee pro6000_supcon_ensemble_result.txt

echo "========== 完成。最终 Score 见上方 '>>> best' 行 =========="
