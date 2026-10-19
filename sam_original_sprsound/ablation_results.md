# SPRSound 事件级 7 分类 · 原版模型消融（2026-10-18/19，RTX 4090）

论文阶梯 **AST → +WRS → +SAM** 在 SPRSound 7 分类上的完整消融。
`train_event.py` 新增 `--no_wrs` / `--no_sam` 开关（默认全开 = 论文最终配置，行为不变）。

## 训练协议（三行完全一致，只切 WRS/SAM）

- 数据：`sprsound_event_7class.npz`（train 6656 / test 2433 事件，inter+intra 合并测试集）
- AST (AudioSet 预训练) + Dropout 0.3 + Linear，CE(label_smoothing=0.1)，lr 1e-5，AdamW，wd 1e-4
- 20 epochs，batch 8，波形级增广（幅度缩放 + 微噪声，p=0.5 each）
- SAM：rho=0.05 两步；WRS：按类频率反比加权采样
- 复现命令（目录 `sam_original_sprsound/` 下）：

```bash
python train_event.py --no_wrs --no_sam --checkpoint_dir ./checkpoints/ast_adamw   # 行①
python train_event.py --no_sam        --checkpoint_dir ./checkpoints/ast_wrs      # 行②
python train_event.py                 --checkpoint_dir ./checkpoints              # 行③（原配置）
python evaluate_split.py checkpoints/ast_adamw/best_model.pth                      # 分层评估
```

## 双口径说明

- **2c**：异常类判成任一异常类即命中（2-class 折叠；= 论文 EVALUATION METRICS 的 TP 定义，引用 Patch-Mix §3.4 2-class 段 / Nguyen & Pernkopf [9]）
- **4c**：异常类需精确命中（Patch-Mix 公式 (1)）
- 两种口径并列打印；best checkpoint 按 2c 选择（与历史行为一致）

## 结果

### inter-only（论文口径，1429 事件，无泄露）

| 配置 | S_p | S_e(2c) | Score(2c) | S_e(4c) | Score(4c) | Acc |
|---|---|---|---|---|---|---|
| ① AST+CE+AdamW | 88.85 | 96.66 | 92.75 | 83.80 | 86.33 | 87.47 |
| ② AST+WRS+AdamW | 88.46 | 97.94 | **93.20** | 87.15 | **87.80** | 88.10 |
| ③ AST+WRS+SAM（论文配置） | 90.19 | 94.86 | 92.53 | 84.06 | 87.13 | 88.52 |

### intra-only（患者内，1004 事件，有泄露）

| 配置 | S_p | S_e(2c) | Score(2c) | S_e(4c) | Score(4c) | Acc |
|---|---|---|---|---|---|---|
| ① | 95.20 | 87.66 | 91.43 | 75.95 | 85.58 | 89.14 |
| ② | 93.75 | 85.76 | 89.75 | 75.63 | 84.69 | 88.05 |
| ③ | 94.33 | 88.92 | 91.63 | 79.75 | 87.04 | 89.74 |

### 合并测试集（inter+intra，2433 事件）

| 配置 | S_p | S_e(2c) | Score(2c) | S_e(4c) | Score(4c) | Acc |
|---|---|---|---|---|---|---|
| ① | 91.38 | 92.62 | 92.00 | 80.28 | 85.83 | 88.16 |
| ② | 90.57 | 92.48 | 91.52 | 81.99 | 86.28 | 88.08 |
| ③ | 91.84 | 92.20 | 92.02 | 82.13 | 86.98 | 89.03 |

## 结论

1. 三行 2c 极差 ≤0.7 分（单次未设种子，噪声量级）——**WRS 略有正贡献，SAM 在 SPRSound 7 类上零到负贡献**；inter-only 最优反而是不含 SAM 的 ②。
2. 与 ICBHI 上 SAM 显著提升形成对照：SPRSound 7 类任务对原版模型的优化器改造不敏感。
3. ③ 的 inter-only 数字与 9 月日志/论文基线（Score 92.53 / Acc 88.52）逐位一致，评估管线可信。
4. 双口径会改变行间排序（inter 行：2c 为 ②>①>③，4c 为 ②>③>①），论文表格必须声明口径。

## 产物

- 训练日志（双口径逐 epoch）：`ablation_ast_adamw.txt`、`ablation_ast_wrs.txt`（③ 见 `train_log.txt`，旧版单口径）
- 分层评估：`eval_inter_intra_row{1,2,3}.txt`
- checkpoint：`checkpoints/ast_adamw/`、`checkpoints/ast_wrs/`（各 345MB，不入库）

---

# 方法 2.5 × SPRSound 7 分类（2026-10-19/20，README 方法 2.5 协议）

**方法 2.5 定义**（README:455）：原版模型 + 四个消融模块（Adam 优化器，无 SAM）。
本实验将其落到 `train_event.py`（新增 `--optimizer {adamw,adam}` / `--weight_decay` 开关），
协议取 README 命令：**Adam, lr 5e-5, wd 1e-6, 无 SAM, bandcl_weight=1.0**；
其余（CE label_smoothing 0.1、WRS 开、波形增广、20 epochs、batch 8）与本文件上半部分三行保持一致，
使两行之差 = 纯四模块效应。

## 复现命令

```bash
# m25_base：四模块关
python train_event.py --no_sam --optimizer adam --lr 5e-5 --weight_decay 1e-6 \
  --bandcl_weight 1.0 --epochs 20 --checkpoint_dir ./checkpoints/m25_base
# m25_full：四模块开
python train_event.py --use_patch_mix --use_same_band_mix --use_band_importance \
  --use_freq_weighted_lam --use_bandcl --bandcl_weight 1.0 --no_sam \
  --optimizer adam --lr 5e-5 --weight_decay 1e-6 --epochs 20 \
  --checkpoint_dir ./checkpoints/m25_full
```

## 结果（inter-only = 论文口径）

| 配置 | S_p | S_e(2c) | Score(2c) | S_e(4c) | Score(4c) | Acc |
|---|---|---|---|---|---|---|
| m25_base（四模块关） | 88.37 | 94.86 | **91.61** | 82.78 | 85.57 | 86.84 |
| m25_full（四模块开） | 87.40 | 94.09 | **90.75** | 80.21 | 83.80 | 85.44 |
| **四模块净效应** | −1.0 | −0.8 | **−0.86** | −2.6 | −1.77 | −1.40 |

intra-only：base 88.32 → full 90.02（+1.70，2c）；合并测试集：89.99 → 90.37（+0.38）。
训练日志：`m25_base.txt`、`m25_full.txt`；分层评估：`eval_inter_intra_m25_*.txt`。

## 结论（SPRSound 7 分类，全部单种子）

1. **四模块在方法 2.5 协议下 inter-only 净 −0.86（2c）/ −1.77（4c）**——与 README 方法2（SAM+模块）Exp 表"+0.06/−0.81"同方向：四模块无可分辨收益；
2. **协议影响大于模块**：A 协议（Adam 5e-5）整体比 AdamW 1e-5 协议低约 1~1.5 分；
3. inter-only 2c 总排名：WRS+AdamW(93.20) > 纯AdamW(92.75) > WRS+SAM(92.53) > m25_base(91.61) > m25_full(90.75)——**原版简单配置全面不劣于带模块/SAM 的复杂配置**；
4. 双口径在 intra 上会翻转结论（m25_full intra +1.70 但 inter −0.86）——论文必须声明口径。
