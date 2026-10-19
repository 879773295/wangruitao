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
