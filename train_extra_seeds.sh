#!/bin/bash
# 快训 seed3/seed4（10ep），为强集成攒成员
PY="C:/Users/diaoqingguang/miniconda3/envs/patchmix_torch201/python.exe"
cd "D:/tmp/class/patch-mix_contrastive_learning" || exit 1

run_seed() {
  local seed=$1
  echo "===== seed${seed}@10ep started $(date +%H:%M:%S) ====="
  $PY -u main.py --method patchmix_cl --epochs 10 --batch_size 8 --num_workers 0 --seed $seed \
    --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
    --from_sl_official --audioset_pretrained --ma_update --ma_beta 0.5 \
    --save_dir ./save --tag torch201_seed${seed}_e10 > seed${seed}_e10.txt 2>&1
  echo "===== seed${seed}@10ep done $(date +%H:%M:%S) ====="
}

run_seed 3
run_seed 4
echo "===== ALL EXTRA SEEDS DONE $(date +%H:%M:%S) ====="
