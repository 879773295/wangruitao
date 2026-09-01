#!/bin/bash
# 小数据快验证矩阵：同一配置，每次只改一个变量，相对排序找超 62 的候选
PY="C:/Users/diaoqingguang/miniconda3/envs/patchmix_torch201/python.exe"
cd "D:/tmp/class/patch-mix_contrastive_learning" || exit 1
export ICBHI_MAX_FILES=160
export PYTHONHASHSEED=0

COMMON="--method patchmix_cl --epochs 20 --batch_size 8 --num_workers 0 --seed 1 \
--optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--from_sl_official --audioset_pretrained \
--save_dir ./save_fast --print_freq 30"

run() {
  local name="$1"; shift
  echo "===== RUN $name started $(date +%H:%M:%S) ====="
  $PY main.py --tag "fast_$name" $COMMON "$@" > "fast_log_$name.txt" 2>&1
  local line=$(grep -E "Best Score|Score:" "fast_log_$name.txt" | tail -3)
  echo "===== RUN $name done $(date +%H:%M:%S) ====="
  echo "$line"
  echo ""
}

run base
run alpha2 --alpha 2.0
run weighted --weighted_loss
run fbe --model ast_fbe --use_freq_band_embed
run timedom --time_domain
run rawaug2 --raw_augment 2
run negdiff --negative_pair diff_label

echo "===== ALL DONE $(date +%H:%M:%S) ====="
