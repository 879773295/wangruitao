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

## 方法 2：权重融合（Score 61.29）

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
