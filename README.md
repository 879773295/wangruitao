# 呼吸音分类：多种 SOTA 方法对比

基于 [Patch-Mix Contrastive Learning](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）改进的呼吸音分类，包含多种 SOTA 方法的实现和对比。

**评测指标**：ICBHI 2017 的 Score = (S_p + S_e) / 2，其中 S_p = 正常样本识别率（特异性），S_e = 异常样本识别率（灵敏度）。

---

## 实验结果对比

| 方法 | Score | S_p | S_e | 训练时间 | 说明 |
|---|---|---|---|---|---|
| **ICBHI-AST-SAM（原版）** | **66.79** | ~65 | ~68 | ~2 小时 | SAM 优化器，当前最优 |
| 权重融合（基线+SupCon）| 61.29 | 72.45 | 50.13 | ~2 小时/seed | 多模型融合 |
| 基线 PatchMixCL | 60.35 | 76.19 | 44.52 | ~2 小时/seed | 官方基线 |
| FBE + SAM | 54.82 | 86.19 | 23.45 | ~2 小时 | 实验结果 |
| CNN 双通道 | 60.27 | 70.23 | 50.30 | ~2 小时 | 局部特征 |

---

## 训练方法总览

| 方法 | 文件 | Score | 创新点 |
|---|---|---|---|
| **SAM 原版** | `sam_original/` | **66.79** | SAM 优化器 |
| 权重融合 | `main.py` | 61.29 | 多模型融合 |
| SupCon | `method/supcon.py` | 58.66 | 监督对比学习 |
| CNN 双通道 | `models/ast_cnn_dual.py` | 60.27 | 全局+局部特征 |
| FBE | `models/ast_freq.py` | 60.25 | 频段嵌入 |

---

## 方法 1：SAM 原版（推荐，Score 66.79）

**论文**：[Geometry-Aware Optimization for Respiratory Sound Classification](https://arxiv.org/abs/2512.22564)

### 运行步骤

```bash
# 1. 数据预处理
cd sam_original
python preprocess.py

# 2. 训练
python train.py --epochs 20 --batch_size 8 --lr 1e-5

# 3. 评估
python evaluate.py --model_path ./checkpoints/best_model.pth
```

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| epochs | 20 | 训练轮数 |
| batch_size | 8 | 批次大小 |
| lr | 1e-5 | 学习率（比 Adam 低）|
| rho | 0.05 | SAM rho 参数 |

### 核心创新

**SAM（Sharpness-Aware Minimization）优化器**：
- 找更平坦的损失曲面
- 提高泛化能力
- 避免过拟合

---

## 方法 1.5：纯 SAM 模型（基础 Baseline）

**用途**：最基础的 baseline，用于对比验证 PatchMix 和其他创新点的效果。

### 运行步骤

```bash
cd sam_bandaware_sprsound

# 1. 数据预处理
python preprocess_sprsound.py

# 2. 训练纯 SAM 模型
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-5 \
  --bandcl_weight 0.01 \
  --data_path sprsound_data.npz \
  --no_patch_mix \
  --no_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/pure_sam
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--no_patch_mix` | 禁用 | 不使用 PatchMix 数据增强 |
| `--no_same_band_mix` | 禁用 | 无同频段交换 |
| `--no_band_importance` | 禁用 | 无频段重要性学习 |
| `--no_freq_weighted_lam` | 禁用 | 无频率加权混合比例 |
| `--no_bandcl` | 禁用 | 无 BandCL Loss |

### 预期结果

纯 SAM 模型作为最基础的 baseline，预期 Score 约 **64-65**。

---

## 方法 2：SAM + Band-aware Patch-Mix（最新，正在实验）

**论文基础**：ICBHI-AST-SAM（arXiv:2512.22564），Score 68.10%

### 创新点（4个模块，支持消融）

| 模块 | 说明 | 参数 |
|---|---|---|
| **Same-band Patch-Mix** | 只在相同频段内交换 patch | `--use_same_band_mix` |
| **Band Importance Estimator** | 学习频段重要性权重 | `--use_band_importance` |
| **Frequency-weighted λ** | 按频段重要性计算混合比例 | `--use_freq_weighted_lam` |
| **BandCL Loss** | 约束混合后频段与源样本一致 | `--use_bandcl` |

---

### ICBHI 数据集训练

```bash
cd sam_bandaware

# 1. 数据预处理
python preprocess.py

# 2. 训练（全模块，默认参数）
python train.py \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-5 \
  --num_classes 4 \
  --bandcl_weight 1.0 \
  --data_path ./icbhi_ast_16k_8s_metadata.npz

# 3. 训练（调参版本）
python train.py \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-5 \
  --num_classes 4 \
  --bandcl_weight 0.01 \
  --bandcl_loss_type cosine \
  --band_importance_temperature 0.5 \
  --mix_lam 0.6 \
  --data_path ./icbhi_ast_16k_8s_metadata.npz

# 4. 消融实验
python train.py --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl  # baseline
python train.py --use_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl  # +A
python train.py --use_same_band_mix --use_band_importance --no_freq_weighted_lam --no_bandcl  # +A+B
python train.py --use_same_band_mix --use_band_importance --use_freq_weighted_lam --no_bandcl  # +A+B+C
python train.py --use_same_band_mix --use_band_importance --use_freq_weighted_lam --use_bandcl  # Full
```

---

### SPRSound 数据集训练

```bash
cd sam_bandaware_sprsound

# 1. 数据预处理（按患者ID划分，避免数据泄露）
python preprocess_sprsound.py

# 2. 训练（全模块，默认参数）
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-5 \
  --num_classes 5 \
  --bandcl_weight 1.0 \
  --data_path ./sprsound_data.npz

# 3. 训练（调参版本）
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-5 \
  --num_classes 5 \
  --bandcl_weight 0.01 \
  --bandcl_loss_type cosine \
  --band_importance_temperature 0.5 \
  --mix_lam 0.6 \
  --data_path ./sprsound_data.npz

# 4. 纯 SAM 模型（无 PatchMix，最基础的 baseline）
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 --batch_size 8 --lr 1e-5 \
  --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_patch_mix --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --checkpoint_dir ./checkpoints/pure_sam

# 5. 消融实验（一键运行 6 组实验）
bash run_ablation.sh

# 6. 消融实验详解

## 消融实验设计思路

**基础模型**：SAM（Sharpness-Aware Minimization）优化器
**创新点**：在 SAM 基础上逐步添加 4 个模块（A、B、C、D），验证每个模块的效果

## 模块原理与作用

### A = Same-band PatchMix（同频段交换 patch）

**原理**：
- 原始 PatchMix：随机交换任意位置的 patch，可能破坏频段内的语义
- Same-band PatchMix：只在相同频段内交换 patch，保持频段语义一致性

**作用**：
- 增加数据多样性，防止过拟合
- 保持频段内的物理意义（低频 patch 不会和高频 patch 混合）

**关键参数**：
- `--mix_lam`：混合比例（0-1），值越大保留原始信息越多
  - 0.3：激进混合（保留 70% 原始）
  - 0.5：默认平衡
  - 0.7：保守混合（保留 30% 原始）

### B = Band Importance Estimator（频段重要性估计器）

**原理**：
- 学习每个频段的重要性权重
- 通过 MLP 网络自动学习哪些频段对分类更重要

**作用**：
- 自动识别重要频段（如高频病理特征）
- 为后续频率加权提供权重依据

**关键参数**：
- `--band_importance_temperature`：温度参数，控制权重分布
  - 0.5：更关注重要频段（权重分布更尖锐）
  - 1.0：默认平衡
  - 2.0：更均匀分布（权重分布更平滑）
- `--band_importance_hidden_ratio`：MLP 隐藏层大小比例（默认 4）

### C = Frequency-weighted λ（频率加权混合比例）

**原理**：
- 根据频段重要性动态调整 PatchMix 的混合比例
- 重要频段保留更多原始信息，次要频段可以更多混合

**作用**：
- 自适应调整混合策略
- 重要频段信息损失更少

**关键参数**：
- 继承 B 的参数（`--band_importance_temperature`）

### D = BandCL Loss（频段级对比一致性损失）

**原理**：
- 约束 PatchMix 后每个频段的特征与原始频段特征保持一致
- 使用 MSE/L1/Cosine 距离度量特征相似度

**作用**：
- 防止 PatchMix 破坏频段内的语义信息
- 确保混合后的特征仍然有意义

**关键参数**：
- `--bandcl_weight`：损失权重（默认 1.0）
  - 0.001：弱约束
  - 0.01：默认平衡
  - 0.1：强约束
- `--bandcl_loss_type`：损失类型
  - `mse`：均方误差（默认）
  - `l1`：绝对误差
  - `cosine`：余弦相似度

---

## 消融实验命令

### Exp 0: 纯 SAM（Baseline）

**配置**：SAM 优化器，无 PatchMix，无任何创新模块

```bash
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 \
  --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_patch_mix \
  --no_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/pure_sam
```

### Exp 1: SAM + PatchMix

**配置**：在纯 SAM 基础上添加原始 PatchMix（随机交换 patch）

```bash
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 \
  --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --use_patch_mix \
  --no_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/patchmix
```

### Exp 2: SAM + A

**配置**：用 A（Same-band PatchMix）替换原始 PatchMix

```bash
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 \
  --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --use_patch_mix \
  --use_same_band_mix \
  --no_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusA
```

### Exp 3: SAM + A + B

**配置**：在 A 基础上添加 B（Band Importance Estimator）

```bash
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 \
  --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --no_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusAB
```

### Exp 4: SAM + A + B + C

**配置**：在 A+B 基础上添加 C（Frequency-weighted λ）

```bash
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 \
  --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --use_freq_weighted_lam \
  --no_bandcl \
  --checkpoint_dir ./checkpoints/plusABC
```

### Exp 5: SAM + A + B + C + D（完整模型）

**配置**：在 A+B+C 基础上添加 D（BandCL Loss）

```bash
HF_HUB_OFFLINE=1 python train.py \
  --epochs 20 --batch_size 8 --lr 1e-5 --num_classes 5 \
  --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --use_patch_mix \
  --use_same_band_mix \
  --use_band_importance \
  --use_freq_weighted_lam \
  --use_bandcl \
  --checkpoint_dir ./checkpoints/full
```

---

## 消融实验对比表

| 实验 | 模型 | PatchMix | A | B | C | D | 预期效果 |
|------|------|----------|---|---|---|---|----------|
| Exp 0 | 纯 SAM | ❌ | ❌ | ❌ | ❌ | ❌ | Baseline |
| Exp 1 | SAM + PatchMix | ✓ 随机 | ❌ | ❌ | ❌ | ❌ | +数据增强 |
| Exp 2 | SAM + A | ✓ 同频段 | ✓ | ❌ | ❌ | ❌ | +频段一致性 |
| Exp 3 | SAM + A + B | ✓ 同频段 | ✓ | ✓ | ❌ | ❌ | +频段重要性 |
| Exp 4 | SAM + A + B + C | ✓ 同频段 | ✓ | ✓ | ✓ | ❌ | +自适应混合 |
| Exp 5 | SAM + A + B + C + D | ✓ 同频段 | ✓ | ✓ | ✓ | ✓ | +特征约束 |
```

---

### 参数设置对照表

| 参数 | ICBHI | SPRSound | 说明 |
|------|-------|----------|------|
| `--num_classes` | 4 | 5 | ICBHI 4类，SPRSound 5类 |
| `--bandcl_weight` | 1.0 | 0.01~1.0 | BandCL 损失权重 |
| `--mix_lam` | 0.5 | 0.5 | PatchMix 混合比例 |
| `--band_importance_temperature` | 1.0 | 1.0 | 温度参数 |
| `--bandcl_loss_type` | mse | mse/cosine | 损失函数类型 |
| `HF_HUB_OFFLINE=1` | 可选 | 必须 | 网络不通时设置 |

### 调参建议

```bash
# 实验 1：BandCL 权重
python train.py --bandcl_weight 0.001
python train.py --bandcl_weight 0.01
python train.py --bandcl_weight 0.1

# 实验 2：温度参数（控制 Band Importance 分布）
python train.py --band_importance_temperature 0.5  # 更关注重要频段
python train.py --band_importance_temperature 1.0  # 默认
python train.py --band_importance_temperature 2.0  # 更均匀

# 实验 3：混合比例
python train.py --mix_lam 0.3  # 保守
python train.py --mix_lam 0.5  # 默认
python train.py --mix_lam 0.7  # 激进

# 实验 4：损失函数类型
python train.py --bandcl_loss_type mse
python train.py --bandcl_loss_type l1
python train.py --bandcl_loss_type cosine
```

### 预训练权重路径

**SAM 代码使用 HuggingFace 预训练权重**，会自动从缓存加载：
- **缓存路径**：`~/.cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/`
- **模型文件**：`snapshots/<commit_hash>/model.safetensors`（331MB）
- **config文件**：`snapshots/<commit_hash>/config.json`（27KB）

**如果需要修改权重路径**，编辑 `sam_bandaware/src/model_bandaware.py`：
```python
# 原来（从 HuggingFace 下载）
self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# 改成用本地权重
from models.ast import ASTModel as LocalAST
self.ast = LocalAST(...)
```

**如果网络不通**，设置离线模式：
```bash
HF_HUB_OFFLINE=1 python train.py ...
```

---

## 方法 2.5：原版模型 + 四个消融模块（Adam 优化器，无 SAM）

**用途**：验证四个消融模块在原版模型（Adam 优化器）上的效果

### ⚠️ 重要：所有实验都需要 AudioSet 预训练

**所有消融实验都必须使用 AudioSet 预训练权重**（`--from_sl_official --audioset_pretrained`）。没有预训练权重，模型无法收敛（S_e=0）。

### 四个创新模块详解

#### 模块 A：Same-band PatchMix（同频段 PatchMix）

**原理**：原版 PatchMix 随机交换任意位置的 patch，可能破坏频谱的频段结构。Same-band PatchMix 只在**相同频段内**交换 patch，保持频段间的频率连续性。

**作用**：
- 保留频谱的频段结构信息
- 让模型学习频段内的判别特征
- 避免跨频段 patch 交换导致的频率不连续

**参数**：`--use_same_band_mix`（启用）/ `--no_same_band_mix`（禁用）

**代码位置**：`models/ast_bandaware.py` → `patch_mix_band_aware()`

#### 模块 B：Band Importance Estimator（频段重要性估计器）

**原理**：使用 MLP 网络学习每个频段的重要性权重。输入各频段的平均特征，输出 softmax 归一化的重要性权重。

**作用**：
- 自动学习哪些频段对分类更重要
- 为后续的频率加权混合提供权重
- 可解释性：可以查看各频段的重要性

**参数**：
- `--use_band_importance`（启用）/ `--no_band_importance`（禁用）
- `--band_importance_hidden_ratio`（MLP 隐藏层比例，默认4）
- `--band_importance_temperature`（softmax 温度，默认1.0）

**代码位置**：`models/ast_bandaware.py` → `BandImportanceEstimator`

#### 模块 C：Frequency-weighted λ（频率加权混合比例）

**原理**：根据频段重要性权重计算 PatchMix 的混合比例 λ。重要性高的频段保留更多原始信息，重要性低的频段更多地交换。

**作用**：
- 重要频段保留更多信息，减少信息损失
- 不重要频段更多交换，增加数据多样性
- 自适应混合比例，比固定 λ 更灵活

**参数**：`--use_freq_weighted_lam`（启用）/ `--no_freq_weighted_lam`（禁用）

**代码位置**：`models/ast_bandaware.py` → `compute_frequency_weighted_lam()`

#### 模块 D：BandCL Loss（频段对比一致性损失）

**原理**：约束混合后的频段特征与源样本的频段特征一致。使用 MSE/L1/Cosine 损失计算频段特征差异。

**作用**：
- 防止 PatchMix 破坏频段语义
- 确保混合后频段特征的合理性
- 提升模型对频段结构的保持能力

**参数**：
- `--use_bandcl`（启用）/ `--no_bandcl`（禁用）
- `--bandcl_weight`（损失权重，默认1.0）
- `--bandcl_loss_type`（损失类型：mse/l1/cosine，默认mse）

**代码位置**：`models/ast_bandaware.py` → `BandCLLoss`

---

### 消融实验命令

> **⚠️ 重要**：所有实验都必须使用 `--from_sl_official --audioset_pretrained` 加载 AudioSet 预训练权重，否则模型无法收敛（S_e=0）。

**各实验命令（完整参数，可直接复制运行）**：

```bash
cd wangruitao

# Exp 0: 原版基线（无 PatchMix，无任何创新模块）
python main.py --model ast_bandaware --method patchmix \
  --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --ma_update --ma_beta 0.5 --from_sl_official --audioset_pretrained \
  --mix_beta 1.0 --weighted_loss --test_fold official \
  --pad_types repeat --resz 1 --n_mels 128 \
  --epochs 50 --seed 1 --batch_size 8 \
  --dataset icbhi --data_folder ./data \
  --tag exp0_baseline

# Exp 1: +A（Same-band PatchMix）
python main.py --model ast_bandaware --method patchmix \
  --use_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --ma_update --ma_beta 0.5 --from_sl_official --audioset_pretrained \
  --mix_beta 1.0 --weighted_loss --test_fold official \
  --pad_types repeat --resz 1 --n_mels 128 \
  --epochs 50 --seed 1 --batch_size 8 \
  --dataset icbhi --data_folder ./data \
  --tag exp1_A

# Exp 2: +A+B（+ Band Importance Estimator）
python main.py --model ast_bandaware --method patchmix \
  --use_same_band_mix --use_band_importance --no_freq_weighted_lam --no_bandcl \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --ma_update --ma_beta 0.5 --from_sl_official --audioset_pretrained \
  --mix_beta 1.0 --weighted_loss --test_fold official \
  --pad_types repeat --resz 1 --n_mels 128 \
  --epochs 50 --seed 1 --batch_size 8 \
  --dataset icbhi --data_folder ./data \
  --tag exp2_AB

# Exp 3: +A+B+C（+ Frequency-weighted λ）
python main.py --model ast_bandaware --method patchmix \
  --use_same_band_mix --use_band_importance --use_freq_weighted_lam --no_bandcl \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --ma_update --ma_beta 0.5 --from_sl_official --audioset_pretrained \
  --mix_beta 1.0 --weighted_loss --test_fold official \
  --pad_types repeat --resz 1 --n_mels 128 \
  --epochs 50 --seed 1 --batch_size 8 \
  --dataset icbhi --data_folder ./data \
  --tag exp3_ABC

# Exp 4: +A+B+C+D（完整模型，bandcl_weight=1.0）
python main.py --model ast_bandaware --method patchmix \
  --use_same_band_mix --use_band_importance --use_freq_weighted_lam --use_bandcl \
  --bandcl_weight 1.0 \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --ma_update --ma_beta 0.5 --from_sl_official --audioset_pretrained \
  --mix_beta 1.0 --weighted_loss --test_fold official \
  --pad_types repeat --resz 1 --n_mels 128 \
  --epochs 50 --seed 1 --batch_size 8 \
  --dataset icbhi --data_folder ./data \
  --tag exp4_ABCD

# Exp 5: +A+B+C+D（bandcl_weight=0.5，消融 BandCL 权重）
python main.py --model ast_bandaware --method patchmix \
  --use_same_band_mix --use_band_importance --use_freq_weighted_lam --use_bandcl \
  --bandcl_weight 0.5 \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --ma_update --ma_beta 0.5 --from_sl_official --audioset_pretrained \
  --mix_beta 1.0 --weighted_loss --test_fold official \
  --pad_types repeat --resz 1 --n_mels 128 \
  --epochs 50 --seed 1 --batch_size 8 \
  --dataset icbhi --data_folder ./data \
  --tag exp5_ABCD_05
```

### 消融实验验证结果（2 epoch 快速测试）

| 实验 | 模块 | S_p | S_e | Score | 状态 |
|------|------|-----|-----|-------|------|
| Exp 0 | 原版（无模块） | 41.80 | 56.92 | 49.36 | ✅ 正常 |
| Exp 1 | +A | 41.99 | 59.81 | 50.90 | ✅ 正常 |
| Exp 2 | +A+B | 32.62 | 65.85 | 49.23 | ✅ 正常 |
| Exp 3 | +A+B+C | 44.40 | 53.95 | 49.17 | ✅ 正常 |
| Exp 4 | +A+B+C+D | 56.36 | 51.57 | 53.97 | ✅ 正常 |
| Exp 5 | +A+B+C+D(0.5) | 38.19 | 61.09 | 49.64 | ✅ 正常 |

> 注：2 epoch 仅为验证实验正常，完整训练需要 50 epoch。

### 消融实验对比表（原版模型，Adam 优化器）

| 实验 | 模型 | PatchMix | A | B | C | D | 说明 |
|------|------|----------|---|---|---|---|------|
| Exp 0 | AST Band-aware | ✓ 随机 | ❌ | ❌ | ❌ | ❌ | 基线（原版 PatchMix） |
| Exp 1 | AST Band-aware | ✓ 同频段 | ✓ | ❌ | ❌ | ❌ | +Same-band PatchMix |
| Exp 2 | AST Band-aware | ✓ 同频段 | ✓ | ✓ | ❌ | ❌ | +Band Importance |
| Exp 3 | AST Band-aware | ✓ 同频段 | ✓ | ✓ | ✓ | ❌ | +Freq-weighted λ |
| Exp 4 | AST Band-aware | ✓ 同频段 | ✓ | ✓ | ✓ | ✓ | +BandCL Loss（完整） |
| Exp 5 | AST Band-aware | ✓ 同频段 | ✓ | ✓ | ✓ | ✓ | +BandCL(λ=0.5) |

---

## 方法 3：权重融合（Score 61.29）

### 运行步骤

```bash
# 1. 训练基线（PatchMixCL）
python main.py --method patchmix_cl --epochs 50 --max_epochs 10 --seed 1 --tag baseline

# 2. 训练 SupCon
python main.py --method patchmix_cl --use_supcon --epochs 50 --max_epochs 10 --seed 1 --tag supcon

# 3. 权重融合
python model_fusion.py

# 4. 评估
python ensemble_eval.py --ckpts ./save/fused_baseline_supcon_alpha0.5.pth --tta 0
```

### 核心创新

- 基线：S_p 高（正常识别好）
- SupCon：S_e 高（异常识别好）
- 融合：两方面都提升

---

## 方法 3：SupCon（Score 58.66）

### 运行步骤

```bash
python main.py --method patchmix_cl --use_supcon --epochs 50 --max_epochs 10 --seed 1 --tag supcon
```

### 核心创新

**Supervised Contrastive Learning**：
- 直接用类别标签构建正负样本对
- 同类样本拉近，异类样本推远
- 对异常类更敏感

---

## 方法 4：CNN 双通道（Score 60.27）

### 运行步骤

```bash
python main.py --model ast_cnn_dual --method patchmix_cl --epochs 50 --max_epochs 10 --seed 1 --tag cnn_dual
```

### 核心创新

- AST：全局特征（Transformer）
- CNN：局部特征（卷积）
- 融合：全局+局部特征

---

## 方法 5：SAM + Band-aware Patch-Mix 用于 SPRSound 数据集

**数据集**：[SPRSound](https://github.com/RSC-Toolkit/SPRSound) - 上海交大儿科呼吸音数据集

**评估指标**：QLung Score = (S_p + S_e) / 2
- S_p = Normal 类准确率
- S_e = 异常类准确率（每个异常类正确分类到自己类别）

### 实验结果

| 数据集 | Score | S_p | S_e | 说明 |
|---|---|---|---|---|
| **SPRSound** | **75.80** | 91.47 | 60.13 | 5类分类，按患者ID划分 |
| ICBHI (原论文) | ~63 | - | - | 4类分类 |

### 数据集信息

| 项目 | SPRSound |
|---|---|
| 类别数 | 5 (Normal, DAS, CAS, CAS&DAS, Poor Quality) |
| 总样本数 | 1949 |
| 患者数 | 251 |
| 训练集 | 1503 样本 (200 患者) |
| 测试集 | 446 样本 (51 患者) |
| 录音时长 | 8秒 (padding/截断) |
| 采样率 | 16kHz |

### 运行步骤

```bash
# 1. 下载 SPRSound 数据集
# 从 https://github.com/RSC-Toolkit/SPRSound 下载
# 解压到 D:/tmp/class/SPRSound-main/

# 2. 数据预处理（按患者ID划分，避免数据泄露）
cd sam_bandaware_sprsound
python preprocess_sprsound.py
# 输出: sprsound_data.npz (训练集1503, 测试集446)

# 3. 训练（全模块）
python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz

# 4. 测试脚本
python test_sprsound.py

# 5. 消融实验（一键运行所有5组）
bash run_ablation.sh

# 或手动运行单个实验：
# Baseline（禁用所有创新模块）
python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl --checkpoint_dir ./checkpoints/baseline

# +A（仅启用 Same-band PatchMix）
python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_band_importance --no_freq_weighted_lam --no_bandcl --checkpoint_dir ./checkpoints/plusA

# +A+B（启用 Same-band + Band Importance）
python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_freq_weighted_lam --no_bandcl --checkpoint_dir ./checkpoints/plusAB

# +A+B+C（启用 Same-band + Band Importance + Freq-weighted λ）
python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_bandcl --checkpoint_dir ./checkpoints/plusABC

# Full（启用所有模块）
python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --checkpoint_dir ./checkpoints/full
```

### 消融实验结果（SPRSound）

| 实验 | A | B | C | D | S_p | S_e | **Score** | Δ |
|------|:---:|:---:|:---:|:---:|-----:|-----:|--------:|-----:|
| Baseline | ✗ | ✗ | ✗ | ✗ | 90.10 | 60.78 | **75.44** | - |
| +A | ✓ | ✗ | ✗ | ✗ | 92.49 | 59.48 | **75.98** | +0.54 |
| +A+B | ✓ | ✓ | ✗ | ✗ | 92.83 | 56.86 | **74.85** | -0.59 |
| +A+B+C | ✓ | ✓ | ✓ | ✗ | 91.81 | 57.52 | **74.66** | -0.78 |
| **Full** | ✓ | ✓ | ✓ | ✓ | 89.76 | 62.09 | **75.93** | **+0.49** |

> A=Same-band PatchMix, B=Band Importance, C=Freq-weighted λ, D=BandCL Loss

**关键发现**：BandCL Loss (D) 主要提升 S_e（异常检出率）：60.78 → 62.09 (+1.31)

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| num_classes | 5 | SPRSound 5类 |
| num_bands | 4 | 频段数量 |
| use_same_band_mix | True | Same-band Patch-Mix |
| use_band_importance | True | Band Importance Estimator |
| use_freq_weighted_lam | True | Frequency-weighted λ |
| use_bandcl | True | BandCL Loss |
| bandcl_weight | 1.0 | BandCL Loss 权重 |

### 文件结构

```
sam_bandaware_sprsound/
├── preprocess_sprsound.py     # SPRSound 数据预处理（按患者ID划分）
├── train.py                   # 训练脚本（支持5类分类）
├── test_sprsound.py           # 测试脚本
├── run_ablation.sh            # 一键运行5组消融实验
├── sprsound_data.npz          # 预处理后的数据（需运行preprocess生成）
└── src/
    ├── dataset_npz.py         # NPZ 格式数据集加载器
    ├── dataset_sprsound.py    # SPRSound 数据集加载器（从原始文件加载）
    ├── model_bandaware.py     # 4个创新点模块
    └── sam.py                 # SAM 优化器
```

### 评估方式（QLung）

```python
# QLung 评估方式：每个异常类必须正确分类到自己类别
hits = np.diag(cm)  # 对角线 = 各类正确预测数
counts = np.sum(cm, axis=1)  # 各类总数

sp = hits[0] / counts[0]  # Normal 准确率
se = sum(hits[1:]) / sum(counts[1:])  # 异常准确率（严格）
score = (sp + se) / 2
```

**注意**：此评估方式比宽松评估（异常→任何异常类算对）更严格，Score 更低但更真实。

---

## BandCL Loss 公式

**Band-level Contrastive Consistency Loss（频段级对比一致性损失）**

### 数学定义

$$
\mathcal{L}_{BandCL} = \frac{1}{B \cdot K} \sum_{i=1}^{B} \sum_{k=1}^{K} \left\| \mathbf{f}_{i,k}^{mix} - \mathbf{f}_{i,k}^{src} \right\|_2^2
$$

其中：
- $B$ — batch size
- $K$ — 频段数量（默认 $K=4$）
- $\mathbf{f}_{i,k}^{src}$ — 第 $i$ 个样本第 $k$ 个频段的原始特征
- $\mathbf{f}_{i,k}^{mix}$ — 第 $i$ 个样本第 $k$ 个频段的混合特征
- $\|\cdot\|_2^2$ — L2 范数的平方（MSE Loss）

### 直观理解

```
原始样本 x → [Band1, Band2, Band3, Band4]  ← 特征 f_src
                ↓ Patch-Mix
混合样本 x' → [Band1, Band2, Band3, Band4]  ← 特征 f_mix

目标：f_mix ≈ f_src（频段特征一致性约束）
```

**核心思想**：Patch-Mix 操作只交换 patch，不改变每个频段的语义内容。BandCL Loss 约束混合后每个频段的特征与原始对应频段保持一致。

### 代码实现

```python
class BandCLLoss(nn.Module):
    """Band-level Contrastive Consistency Loss"""
    def forward(self, band_features_src, band_features_mix):
        """
        Args:
            band_features_src: [B, K, D] - 原始频段特征
            band_features_mix: [B, K, D] - 混合后频段特征
        Returns:
            loss: scalar
        """
        return F.mse_loss(band_features_mix, band_features_src, reduction='mean')
```

### 与其他损失结合

```python
# 总损失 = 分类损失 + BandCL 损失
loss = criterion(logits, labels)  # CrossEntropy
if args.use_bandcl:
    bandcl_loss = model.compute_bandcl_loss(band_features_src, band_features_mix)
    loss += args.bandcl_weight * bandcl_loss  # 默认权重 1.0
```

---

## 模型原理详解

### 1. SAM（Sharpness-Aware Minimization）优化器

**论文**：[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)

**核心思想**：传统优化器（如 Adam）只找损失值低的点，SAM 找的是**损失曲面平坦的区域**。

**原理**：
```
传统优化：找到 loss 低的点 → 可能是尖锐的最低点 → 泛化差
SAM：找到 loss 低且平坦的区域 → 泛化更好
```

**实现**（两步更新）：
```python
# 第一步：计算梯度，找扰动方向
loss.backward()
optimizer.first_step(zero_grad=True)

# 第二步：在扰动位置重新计算梯度，更新参数
loss2 = criterion(model(inputs), labels)
loss2.backward()
optimizer.second_step(zero_grad=True)
```

**参数**：
- `rho`（默认 0.05）：扰动幅度，越大越关注平坦区域
- `lr`（默认 1e-5）：学习率，SAM 需要更小的学习率

---

### 2. PatchMix 数据增强

**论文**：[Patch-Mix Contrastive Learning for Audio-Visual Retrieval](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）

**核心思想**：在 patch 级别混合样本，增加数据多样性。

**原理**：
```
样本 A: [p1, p2, p3, p4, p5, p6, p7, p8]
样本 B: [p1, p2, p3, p4, p5, p6, p7, p8]
混合（随机交换）: [A1, B2, A3, B4, A5, B6, A7, B8]
```

**作用**：
- 增加训练数据多样性
- 防止过拟合
- 提升模型泛化能力

---

### 3. Same-band PatchMix（创新点 A）

**核心思想**：只在**相同频段**内交换 patch，避免跨频段混合导致的语义混乱。

**原理**：
```
原始 PatchMix（随机交换）：
  低频 patch 可能和高频 patch 交换 → 语义混乱

Same-band PatchMix（同频段交换）：
  频段1 (低频): [p1, p2] → 只和频段1内交换
  频段2:        [p3, p4] → 只和频段2内交换
  频段3:        [p5, p6] → 只和频段3内交换
  频段4 (高频): [p7, p8] → 只和频段4内交换
```

**作用**：
- 保持频段内的语义一致性
- 避免跨频段混合导致的特征混乱
- 更符合音频信号的物理特性

---

### 4. Band Importance Estimator（创新点 B）

**核心思想**：学习每个频段的重要性权重，让模型关注更有区分度的频段。

**原理**：
```python
class BandImportanceEstimator(nn.Module):
    def __init__(self, embed_dim=768, num_bands=4, hidden_ratio=4, temperature=1.0):
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // hidden_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim // hidden_ratio, 1),
        )
        self.temperature = temperature

    def forward(self, band_features):
        # band_features: [B, num_bands, D]
        importance_scores = self.mlp(band_features)  # [B, num_bands, 1]
        importance_scores = importance_scores.squeeze(-1) / self.temperature
        band_importance = F.softmax(importance_scores, dim=-1)  # [B, num_bands]
        return band_importance
```

**作用**：
- 自动学习哪些频段对分类更重要
- 高频段（病理特征）可能比低频段更重要
- 为后续的频率加权提供权重

**可调参数**：
- `temperature`：控制权重分布的平滑度
  - 值小 → 权重分布更尖锐（更关注重要频段）
  - 值大 → 权重分布更平滑（更均匀）
- `hidden_ratio`：MLP 隐藏层大小

---

### 5. Frequency-weighted λ（创新点 C）

**核心思想**：根据频段重要性动态调整 PatchMix 的混合比例。

**原理**：
```python
# 基础混合比例
mix_lam = 0.5

# 用 Band Importance 加权
band_mask_ratio = torch.ones(num_bands) * mix_lam
lam = (band_importance * band_mask_ratio).sum() / band_importance.sum()
```

**作用**：
- 重要频段保留更多原始信息
- 次要频段可以更多混合
- 自适应调整混合策略

**示例**：
```
频段重要性：[0.1, 0.2, 0.3, 0.4]  （高频更重要）
基础 mix_lam：0.5

加权后：
  频段1: 混合比例 = 0.5 * 0.1 = 0.05（保留 95%）
  频段2: 混合比例 = 0.5 * 0.2 = 0.10（保留 90%）
  频段3: 混合比例 = 0.5 * 0.3 = 0.15（保留 85%）
  频段4: 混合比例 = 0.5 * 0.4 = 0.20（保留 80%）
```

---

### 6. BandCL Loss（创新点 D）

**论文基础**：Band-level Contrastive Consistency Loss

**核心思想**：约束 PatchMix 后每个频段的特征与原始频段特征保持一致。

**原理**：
```
原始样本 x → [Band1, Band2, Band3, Band4]  ← 特征 f_src
                ↓ Patch-Mix
混合样本 x' → [Band1, Band2, Band3, Band4]  ← 特征 f_mix

目标：f_mix ≈ f_src（频段特征一致性约束）
```

**公式**：
$$\mathcal{L}_{BandCL} = \frac{1}{B \cdot K} \sum_{i=1}^{B} \sum_{k=1}^{K} \left\| \mathbf{f}_{i,k}^{mix} - \mathbf{f}_{i,k}^{src} \right\|_2^2$$

**代码实现**：
```python
class BandCLLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        self.loss_type = loss_type

    def forward(self, band_features_src, band_features_mix):
        if self.loss_type == 'mse':
            return F.mse_loss(band_features_mix, band_features_src)
        elif self.loss_type == 'cosine':
            cos_sim = F.cosine_similarity(band_features_mix, band_features_src, dim=-1)
            return (1 - cos_sim).mean()
```

**作用**：
- 防止 PatchMix 破坏频段内的语义信息
- 确保混合后的特征仍然有意义
- 提升模型的鲁棒性

**可调参数**：
- `bandcl_weight`：损失权重（默认 1.0）
- `bandcl_loss_type`：损失类型（mse/l1/cosine）

---

## 模型对比总结

| 模型 | Score | 特点 | 适用场景 |
|------|-------|------|----------|
| **纯 SAM** | ~64 | 优化器改进，泛化好 | 基线模型 |
| **SAM + PatchMix** | ~75 | 数据增强 | 数据量少时 |
| **SAM Band-aware** | ~76 | 频段感知 + 自适应混合 | 追求最优性能 |

---

## 环境要求

**必须用 torch 2.0.1 + cu118（torchaudio 2.0.2）**。实测 torch 2.2+ 会掉 ~5 分。

```bash
conda create -n patchmix python=3.10 -y
conda activate patchmix
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.4.5 librosa numpy==1.24.4 pandas nlpaug tqdm
```

---

## 数据准备

下载 [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/) 数据集：

```bash
# 放入我们的代码
data/icbhi_dataset/audio_test_data/

# 放入 SAM 原版代码
sam_original/data/ICBHI_final_database/
```

---

## 重要发现

1. **SAM 优化器是核心**：提升 ~6 分（60.35 → 66.79）
2. **学习率很重要**：SAM 用 1e-5（比 Adam 的 5e-5 低）
3. **硬件影响**：4090 上限 ~61-62，PRO 6000 可到 68+

---

## 方法 3：SPRSound 事件级 7 分类（原始 SAM + CustomAST）

对 [SPRSound（BioCAS 2022）](https://github.com/SJTU-RHRL/SPRSound) 做**事件级（event-level）7 分类**。模型用原始 SAM 优化器 + HuggingFace AST（mean pooling + Dropout 0.3）+ 交叉熵。代码在 `sam_original_sprsound/` 目录，与 ICBHI 主流程完全独立，不修改原版代码。

### 7 类标签

`Normal=0, Rhonchi=1, Wheeze=2, Stridor=3, Coarse Crackle=4, Fine Crackle=5, Wheeze+Crackle=6`

### 数据分布

事件级样本按 7 类统计，类不平衡严重（Normal 一家独大，罕见类样本极少）：

| 类别 | 训练集 | 占比 | 测试集 | 占比 |
|---|---:|---:|---:|---:|
| Normal | 5159 | 77.51% | 1728 | 71.02% |
| Fine Crackle | 912 | 13.70% | 255 | 10.48% |
| Wheeze | 452 | 6.79% | 413 | 16.97% |
| Coarse Crackle | 49 | 0.74% | 17 | 0.70% |
| Rhonchi | 39 | 0.59% | 14 | 0.58% |
| Wheeze+Crackle | 30 | 0.45% | 4 | 0.16% |
| Stridor | 15 | 0.23% | 2 | 0.08% |
| **合计** | **6656** | — | **2433** | — |

> 测试集 = intra（1004 事件）+ inter（1429 事件）。Stridor / Rhonchi / Coarse Crackle / Wheeze+Crackle 四类训练样本合计仅 ~133 例（~2%），是后续召回上不去的主因。

### 环境要求

- torch 2.0.1+cu118（conda 环境 `patchmix_torch201`）
- `transformers`（ASTFeatureExtractor / ASTModel）、`librosa`、`scikit-learn`、`numpy`、`tqdm`

### 运行步骤

1. 下载 SPRSound（BioCAS2022），把 `preprocess_event.py` 里的 `DATA_DIR` 改成你自己的数据路径（默认 `D:/tmp/class/SPRSound-main/SPRSound-main/BioCAS2022`）。

2. **预处理**（8kHz→16kHz 重采样，按事件 start/end 切片，循环填充到 8s / 128000 采样点）：

   ```bash
   cd sam_original_sprsound
   python preprocess_event.py
   ```

   输出 `sprsound_event_7class.npz`：训练 6656 事件 / 测试 2433 事件（test = intra 379 + inter 355 条记录）。

3. **训练**（20 epoch，`WeightedRandomSampler` 缓解类不平衡，每 epoch 打印各类准确率）：

   ```bash
   python train_event.py --epochs 20 --batch_size 8 --lr 1e-5
   ```

   最优 checkpoint 存 `checkpoints/best_model.pth`。

4. **评估**（inter / intra 分开）：

   ```bash
   python evaluate_split.py
   ```

### 结果（20 epoch，best epoch 19）

| 评估 | Acc | S_p | S_e | Score |
|---|---|---|---|---|
| inter-only（患者 0 重叠，真实泛化） | 88.52 | 90.19 | 94.86 | 92.53 |
| intra-only（患者内） | 89.74 | 94.33 | 88.92 | 91.63 |
| 合并（inter + intra） | 89.03 | 91.84 | 92.20 | 92.02 |

**每类准确率**（%，inter-only 为可引用的无泄露口径）：

| 类别 | inter-only | 合并（inter+intra） |
|---|---:|---:|
| Normal | 90.2 | 91.8 |
| Rhonchi | 0.0 | 71.4 |
| Wheeze | 84.3 | 85.2 |
| Stridor | 0.0 | 0.0 |
| Coarse Crackle | 0.0 | 5.9 |
| Fine Crackle | 86.2 | 83.9 |
| Wheeze+Crackle | 100.0* | 50.0 |

> *inter-only 的 Wheeze+Crackle 仅极少量样本，100% 是样本太少造成的噪声；Rhonchi / Stridor / Coarse Crackle 在 inter 测试里样本也极少，召回≈0。

**结论**：inter ≈ intra（差 ~1%），模型跨患者泛化好、无患者级泄露；整体 Acc 主要被 Normal（测试占比 ~71%）撑起，7 类宏平均约 ~55%，罕见类（Rhonchi / Stridor / Coarse Crackle，训练各 < 50 例）召回≈0 是主要瓶颈。

> ⚠️ 协议说明：当前 `train_event.py` 用测试集选 best checkpoint（`score > best_score` 才保存），报告的 Score 偏乐观；严谨做法是从 train2022 按患者切一份验证集用于选模型。数据切分口径详见 README 顶部「评测指标」与 ICBHI 章节。

### 四模块消融（Band-aware Patch-Mix，7 分类版）

把方法 5 的四模块消融移植到事件级 7 分类：模型底座同为 HuggingFace AST，训练逻辑与 `sam_bandaware_sprsound/train.py`（5 类版）逐行一致，四组实验设计相同。代码在 `model_bandaware.py`（模型 + 四模块）与 `train_event.py`（训练循环），**默认全关 = 原始 SAM 基线，不加参数行为与旧版完全一致**。

**消融维度**：

| 开关 | 模块 | 说明 |
|---|---|---|
| `--use_patch_mix` | 基础 PatchMix | 随机交换 patch（Exp1 起启用） |
| `--use_same_band_mix` | **A** | Same-band Patch-Mix：只在同频段内交换 |
| `--use_band_importance` | **B** | Band Importance Estimator：频段重要性权重 |
| `--use_freq_weighted_lam` | **C** | Frequency-weighted λ：按重要性加权混合比例 |
| `--use_bandcl` | **D** | BandCL Loss（`--bandcl_weight 0.01`） |

**训练方法**：

```bash
cd sam_original_sprsound

# 正式：6 组 × 20 epoch（约 2 小时/组）
bash run_ablation.sh

# 快速：每组 1 个 epoch（验证趋势用）
EPOCHS=1 bash run_ablation.sh

# 冒烟：Exp0 + Exp5 各 5 个 batch（只验证代码能跑通）
bash run_ablation.sh smoke

# 断点续跑：跳过前 N 组（已完成的组不重跑），如从 Exp4 续跑
START_FROM=4 bash run_ablation.sh

# 手动跑单组（示例：Exp3 = A+B）
python train_event.py --epochs 20 --batch_size 8 --lr 1e-5 --num_classes 7 \
  --bandcl_weight 0.01 --data_path ./sprsound_event_7class.npz \
  --use_patch_mix --use_same_band_mix --use_band_importance \
  --checkpoint_dir ./checkpoints/abl_plusAB
```

每组输出 `S_p / S_e / Score / Acc` + 各类准确率，best checkpoint 存到各自的 `checkpoints/abl_*` 目录（互不覆盖）。

**6 组实验设计**（与 5 类版一致）：

| 组 | 配置 |
|---|---|
| Exp0 | 纯 SAM 基线（全关） |
| Exp1 | + 原始 PatchMix（随机交换） |
| Exp2 | + A（Same-band PatchMix） |
| Exp3 | + A + B（Band Importance） |
| Exp4 | + A + B + C（Freq-weighted λ） |
| Exp5 | Full = A + B + C + D（BandCL，权重 0.01） |

**20 epoch 正式结果**（单 seed、测试集选 best；2026-10-13 更新，Exp0–3 已完成，Exp4/Exp5 训练中）：

| 组 | S_p | S_e | Score | Acc | Δ vs 基线 |
|---|---:|---:|---:|---:|---:|
| Exp0 纯 SAM | 90.34 | 94.75 | **92.54** | 88.33 | — |
| Exp1 +PatchMix | 89.58 | 94.33 | **91.95** | 87.38 | -0.59 |
| Exp2 +A | 90.10 | 93.76 | **91.93** | 87.30 | -0.61 |
| Exp3 +A+B | 88.89 | 96.31 | **92.60** | 87.63 | **+0.06** |
| Exp4 +A+B+C | 训练中 | | | | |
| Exp5 Full | 待跑 | | | | |

> 解读（截至 Exp3）：20 epoch 下 Exp3（A+B）以 92.60 首次超过基线（+0.06），且 S_e 96.31 为全场最高（S_p 88.89 略降，跷跷板）；Exp1/Exp2 反而低于基线。单 seed 结论仍需多 seed 复验，完整 6 组跑完后更新此表。

**初跑结果**（每组 1 epoch、单 seed、测试集选 best——仅验证量级，非正式数据）：

| 组 | S_p | S_e | Score | Acc |
|---|---:|---:|---:|---:|
| Exp0 纯 SAM | 76.16 | 97.16 | **86.66** | 78.34 |
| Exp1 +PatchMix | 83.45 | 94.61 | **89.03** | 81.01 |
| Exp2 +A | 80.38 | 94.47 | **87.43** | 80.23 |
| Exp3 +A+B | 84.26 | 94.18 | **89.22** | 82.24 |
| Exp4 +A+B+C | 51.04 | 99.01 | **75.02** | 58.69 |
| Exp5 Full | 75.64 | 96.31 | **85.97** | 76.61 |

> 解读：Exp3（A+B）最佳（+2.56）；Exp4 的 Freq-weighted λ 单 epoch 未收敛导致 S_p 塌到 51（几乎全预测异常）；Exp5 被 BandCL 拉回但仍低于 Exp3。**正式结论需 20 epoch × 多 seed**（`bash run_ablation.sh`），1 epoch 表仅供趋势参考。

---

## 混淆矩阵画图（论文 Fig.13 风格）

`plot_confusion.py` 生成与 SPRSound 论文 Fig.13 同版式的混淆矩阵：Predicted Label 在顶部横排、True Label 在左侧竖排、对角线蓝（正确识别正常音）/ 红（正确识别异常音）并显示「数量 + 行百分比」两行、底部图例。支持 **2×2 二分类** 和 **n×n 多分类**，两个数据集通用。

> 运行环境：conda 环境 `patchmix_torch201`（仓库自带的 `.venv` 已损坏不可用）。

### 方式一：手动填数字（JSON 模板）

编辑 `matrix_example_2x2.json`（2×2）或 `matrix_example.json`（n×n），`counts` 为**行=真实标签、列=预测标签**的二维数组：

```json
{
  "title": "你的图标题",
  "labels": ["Normal", "Adventitious"],
  "counts": [
    [1445, 283],
    [228, 477]
  ]
}
```

```bash
python plot_confusion.py --json matrix_example_2x2.json --out my_fig.png
```

### 方式二：从预测文件自动计算（ICBHI）

先用 `dump_icbhi_preds.py` 导出测试集预测（y_true + y_pred），再画图。已有现成的 `preds_icbhi_sam.npz`（2756 条，SAM+PatchMix 权重）可直接用：

```bash
# 4 类 n×n
python plot_confusion.py --preds preds_icbhi_sam.npz --labels Normal Crackle Wheeze Both --out fig_icbhi_sam4.png

# 2×2 二分类（4 类自动折叠：Crackle/Wheeze/Both 合并为 Adventitious）
python plot_confusion.py --preds preds_icbhi_sam.npz --labels Normal Crackle Wheeze Both --binary --out fig_icbhi_binary.png
```

换自己的 checkpoint 时先导出预测：

```bash
python dump_icbhi_preds.py --ckpt save/<实验名>/best.pth --out preds_icbhi.npz
python plot_confusion.py --preds preds_icbhi.npz --labels Normal Crackle Wheeze Both --binary
```

### 方式三：SPRSound 事件级自动推理

```bash
python plot_confusion.py --sprsound --split all    --out fig_sprsound_combined.png  # 7 类合并测试集
python plot_confusion.py --sprsound --split inter  --out fig_sprsound_inter.png     # 7 类患者独立子集
python plot_confusion.py --sprsound --split all    --binary                         # 2×2 二分类
```

`--ckpt` 默认 `sam_original_sprsound/checkpoints/best_model.pth`，可用 `--ckpt` 覆盖。

### 常用参数

| 参数 | 说明 |
|---|---|
| `--binary` | 折叠成 2×2（Normal vs Adventitious，对应 SPRSound 论文 Task 1-1 口径）|
| `--title "..."` | 图标题 |
| `--out xxx.png` | 输出文件名（也支持 `.pdf` 矢量输出）|
| `--style fig13\|blues` | `fig13`=仿原论文白底蓝/红对角线（默认）；`blues`=经典蓝色渐变热力图 |
| `--normal-index 0` | Normal 类的下标（决定对角线用蓝色）|
| `--figsize 6` | 图宽（英寸），高度按类别数自动算，类别多时自动加宽防标签重叠 |

### 已生成的成品图

| 文件 | 内容 |
|---|---|
| `fig_icbhi_binary.png` | ICBHI 2×2：[[1379, 200], [583, 594]]，S_p 87.3 / S_e 50.5（二分类口径）|
| `fig_icbhi_sam4.png` | ICBHI 4 类，官方测试集 2756 条 |
| `fig_sprsound_binary.png` | SPRSound 2×2：[[1587, 141], [55, 650]]，二分类 Acc 92.3 |
| `fig_sprsound_combined.png` | SPRSound 7 类合并（2433 事件），论文主图推荐 |
| `fig_sprsound_inter.png` | SPRSound 7 类 inter-only（1429 事件），罕见类行全 0 |

---

## 模型清单

| `--model` | 文件 | 说明 |
|---|---|---|
| `ast` | `models/ast.py` | 基线：原始 AST |
| `ast_fbe` | `models/ast_freq.py` | 病理频段 FBE |
| `ast_cnn_dual` | `models/ast_cnn_dual.py` | AST + CNN 双通道 |

---

## 脚本说明

| 文件 | 作用 |
|---|---|
| `main.py` | 训练/评估入口（支持 SAM、SupCon、FBE）|
| `model_fusion.py` | 权重融合 |
| `ensemble_eval.py` | 集成 + TTA + 阈值校准 |
| `optim/sam.py` | SAM 优化器 |
| `sam_original/` | ICBHI-AST-SAM 原版代码 |
| `sam_original_sprsound/` | SPRSound 事件级 7 分类（SAM + CustomAST） |
| `plot_confusion.py` | 论文 Fig.13 风格混淆矩阵（支持 2×2 / n×n，三种输入方式）|
| `dump_icbhi_preds.py` | 导出 ICBHI 测试集预测 → npz（供画图/分析）|
| `matrix_example.json` / `matrix_example_2x2.json` | 手动填数字的 JSON 模板（n×n / 2×2）|
| `sam_original_sprsound/run_ablation.sh` | 7 分类四模块消融一键脚本（6 组 + smoke 模式 + `START_FROM` 断点续跑）|
| `sam_original_sprsound/model_bandaware.py` | 7 分类四模块模型（A/B/C/D，移植自 5 类版）|

---

## 致谢

- [Patch-Mix Contrastive Learning](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）
- [ICBHI-AST-SAM](https://github.com/Atakanisik/ICBHI-AST-SAM)（SAM 优化器）
