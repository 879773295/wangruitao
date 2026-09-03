# ICBHI-AST-SAM 原版代码

基于论文：**"Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers"**

- arXiv: 2512.22564
- GitHub: https://github.com/Atakanisik/ICBHI-AST-SAM
- **Score: 68.10%**（Sensitivity 68.31%，Specificity 67.89%）

---

## 核心创新

**SAM（Sharpness-Aware Minimization）优化器**：替代 Adam，找更平坦的损失曲面，提高泛化能力。

---

## 运行步骤

### 1. 环境准备

```bash
pip install torch torchvision torchaudio
pip install transformers==4.30.0
pip install librosa numpy pandas tqdm scikit-learn
```

### 2. 数据准备

```bash
# 下载 ICBHI 数据集
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
unzip ICBHI_final_database.zip -d data/

# 下载 split file
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt
mv ICBHI_challenge_train_test.txt data/
```

### 3. 数据预处理

```bash
python preprocess.py
```

生成 `icbhi_ast_16k_8s_metadata.npz`（约 3.3GB）

### 4. 训练

```bash
python train.py --epochs 20 --batch_size 8 --lr 1e-5
```

**预期结果**：
- Score: ~66-68%
- 训练时间: ~2 小时（4090）

### 5. 评估

```bash
python evaluate.py --model_path ./checkpoints/best_model.pth
```

---

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| epochs | 20 | 训练轮数 |
| batch_size | 8 | 批次大小 |
| lr | 1e-5 | 学习率 |
| rho | 0.05 | SAM rho 参数 |

---

## 模型结构

```
AST (Audio Spectrogram Transformer)
  └─ ImageNet + AudioSet 预训练
  └─ Mean Pooling
  └─ Dropout(0.3)
  └─ Linear(768, 4)
```

---

## SAM 优化器

```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05):
        # 找更平坦的损失曲面
        # 两步更新：first_step + second_step
```

**优势**：
1. 避免过拟合
2. 提高泛化能力
3. 找更鲁棒的最优解

---

## 实验结果（4090）

| Epoch | Loss | Score |
|---|---|---|
| 1 | 1.0278 | 0.62 |
| 4 | 0.6035 | 0.65 |
| 7 | 0.4968 | 0.66 |
| 10 | 0.4346 | 0.66 |
| 20 | 0.3768 | **0.6679** |

---

## 与其他方法对比

| 方法 | Score | 说明 |
|---|---|---|
| 基线 AST（Adam）| 60.35 | 我们之前的结果 |
| 权重融合 | 61.29 | 基线+SupCon |
| **AST + SAM** | **66.79** | 原版代码 |
| 论文报告值 | 68.10 | 论文结果 |

---

## 关键发现

1. **SAM 优化器是核心**：提升 ~6 分（60.35 → 66.79）
2. **不需要改变模型结构**：只换优化器
3. **学习率很重要**：用 1e-5（比 Adam 的 5e-5 低）

---

## 致谢

基于 [ICBHI-AST-SAM](https://github.com/Atakanisik/ICBHI-AST-SAM) 仓库的代码实现。
