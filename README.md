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

# 4. 消融实验（一键运行）
bash run_ablation.sh

# 5. 单独运行消融实验
HF_HUB_OFFLINE=1 python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl --checkpoint_dir ./checkpoints/baseline

HF_HUB_OFFLINE=1 python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --no_band_importance --no_freq_weighted_lam --no_bandcl --checkpoint_dir ./checkpoints/plusA

HF_HUB_OFFLINE=1 python train.py --epochs 20 --batch_size 8 --lr 1e-5 --bandcl_weight 0.01 --data_path sprsound_data.npz \
  --checkpoint_dir ./checkpoints/full
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

---

## 致谢

- [Patch-Mix Contrastive Learning](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）
- [ICBHI-AST-SAM](https://github.com/Atakanisik/ICBHI-AST-SAM)（SAM 优化器）
