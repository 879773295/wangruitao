#!/bin/bash
# 用 50-ep 的 LR 曲线（--epochs 50）提前在 epoch 10 停，拿到高分 seed（同 seed1/1337 条件）
PY="C:/Users/diaoqingguang/miniconda3/envs/patchmix_torch201/python.exe"
cd "D:/tmp/class/patch-mix_contrastive_learning" || exit 1

run_seed() {
  local seed=$1
  echo "===== seed${seed}_s50 started $(date +%H:%M:%S) ====="
  $PY -u main.py --method patchmix_cl --epochs 50 --max_epochs 10 --batch_size 8 --num_workers 0 --seed $seed \
    --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
    --from_sl_official --audioset_pretrained --ma_update --ma_beta 0.5 \
    --save_dir ./save --tag torch201_seed${seed}_s50 > seed${seed}_s50.txt 2>&1
  echo "===== seed${seed}_s50 done $(date +%H:%M:%S) ====="
}

run_seed 5
run_seed 6
echo "===== ALL STRONG SEEDS DONE $(date +%H:%M:%S) ====="
