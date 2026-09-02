# 完整训练流程：基线 + SupCon + 权重融合

本文档详细说明如何训练基线模型、SupCon 模型，以及如何做权重融合和评估。

---

## 目录

1. [环境准备](#1-环境准备)
2. [训练基线模型（PatchMixCL）](#2-训练基线模型patchmixcl)
3. [训练 SupCon 模型](#3-训练-supcon-模型)
4. [权重融合](#4-权重融合)
5. [评估测试](#5-评估测试)
6. [一键脚本](#6-一键脚本)

---

## 1. 环境准备

```bash
# 创建环境
conda create -n patchmix python=3.10 -y
conda activate patchmix

# 安装依赖（必须 torch 2.0.1+cu118）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.4.5 librosa numpy==1.24.4 pandas nlpaug tqdm
```

---

## 2. 训练基线模型（PatchMixCL）

基线模型使用 InfoNCE 对比损失，对异常区域更敏感（S_e 高）。

### 完整训练（50 epoch）

```bash
python main.py \
  --method patchmix_cl \
  --epochs 50 \
  --batch_size 8 \
  --seed 1 \
  --optimizer adam \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --cosine \
  --from_sl_official \
  --audioset_pretrained \
  --ma_update \
  --ma_beta 0.5 \
  --save_dir ./save \
  --tag baseline_seed1
```

**关键参数说明**：
- `--method patchmix_cl`：使用 PatchMix + 对比学习
- `--from_sl_official --audioset_pretrained`：ImageNet + AudioSet 双预训练
- `--ma_update --ma_beta 0.5`：EMA（指数移动平均）
- `--cosine`：余弦退火学习率

**输出**：
- `save/icbhi_ast_patchmix_cl_baseline_seed1/best.pth`

**预期结果**（4090）：
- Score: 60.35
- S_p: 76.19（正常识别好）
- S_e: 44.52（异常识别一般）

### 快训（10 epoch，用于调试）

```bash
python main.py \
  --method patchmix_cl \
  --epochs 50 \
  --max_epochs 10 \
  --batch_size 8 \
  --seed 1 \
  --optimizer adam \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --cosine \
  --from_sl_official \
  --audioset_pretrained \
  --ma_update \
  --ma_beta 0.5 \
  --save_dir ./save \
  --tag baseline_seed1_fast
```

---

## 3. 训练 SupCon 模型

SupCon 使用监督对比损失，直接用类别标签，对正常类更聚焦（S_p 高）。

### 完整训练（50 epoch）

```bash
python main.py \
  --method patchmix_cl \
  --use_supcon \
  --epochs 50 \
  --batch_size 8 \
  --seed 1 \
  --optimizer adam \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --cosine \
  --from_sl_official \
  --audioset_pretrained \
  --ma_update \
  --ma_beta 0.5 \
  --save_dir ./save \
  --tag supcon_seed1
```

**关键参数**：
- `--use_supcon`：使用 SupCon 损失（替代 InfoNCE）

**输出**：
- `save/icbhi_ast_patchmix_cl_supcon_seed1/best.pth`

**预期结果**（4090）：
- Score: 58.66
- S_p: 71.18（正常识别一般）
- S_e: 46.13（异常识别好）

### 快训（10 epoch）

```bash
python main.py \
  --method patchmix_cl \
  --use_supcon \
  --epochs 50 \
  --max_epochs 10 \
  --batch_size 8 \
  --seed 1 \
  --optimizer adam \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --cosine \
  --from_sl_official \
  --audioset_pretrained \
  --ma_update \
  --ma_beta 0.5 \
  --save_dir ./save \
  --tag supcon_seed1_fast
```

---

## 4. 权重融合

把基线和 SupCon 的权重平均，得到一个融合模型。

### 运行融合脚本

```bash
python model_fusion.py
```

**脚本逻辑**：
1. 加载基线模型权重
2. 加载 SupCon 模型权重
3. 平均权重（α=0.5）
4. 保存融合模型

**输出**：
- `save/fused_baseline_supcon_alpha0.5.pth`

**融合公式**：
```
融合权重 = 0.5 × 基线权重 + 0.5 × SupCon权重
```

---

## 5. 评估测试

### 单模型评估

```bash
# 评估基线
python ensemble_eval.py --ckpts ./save/icbhi_ast_patchmix_cl_baseline_seed1/best.pth --tta 0

# 评估 SupCon
python ensemble_eval.py --ckpts ./save/icbhi_ast_patchmix_cl_supcon_seed1/best.pth --tta 0
```

### 融合模型评估

```bash
python ensemble_eval.py --ckpts ./save/fused_baseline_supcon_alpha0.5.pth --tta 0
```

### 多模型集成评估（可选）

```bash
python ensemble_eval.py \
  --ckpts ./save/icbhi_ast_patchmix_cl_baseline_seed1/best.pth \
          ./save/icbhi_ast_patchmix_cl_supcon_seed1/best.pth \
  --tta 16
```

---

## 6. 一键脚本

### PRO 6000 一键运行

```bash
bash pro6000_run.sh
```

**脚本内容**：
1. 快训 8 个 seed 的基线模型（10 epoch）
2. 快训 8 个 seed 的 SupCon 模型（10 epoch）
3. 权重融合（每个 seed 的基线 + SupCon）
4. 集成评估 + TTA + 阈值校准

---

## 结果对比

### 4090 实测结果

| 模型 | Score | S_p | S_e | 说明 |
|---|---|---|---|---|
| 基线（InfoNCE）| **60.35** | 76.19 | 44.52 | S_e 高 |
| SupCon | 58.66 | 71.18 | 46.13 | S_p 高 |
| **权重融合** | **61.29** | 72.45 | 50.13 | **最优** |
| 融合+阈值 | **62.89** | 82.46 | 43.33 | 后处理 |

### 关键发现

1. **权重融合比单模型更好**（+0.94 分）
2. **S_p 和 S_e 是跷跷板关系**——提升一个必然降低另一个
3. **融合 loss 不如权重融合**——因为会损失多样性

---

## 常见问题

**Q：为什么 SupCon 比基线差？**
因为 SupCon 优化的是"类间可分性"，而 ICBHI 的 Score = (S_p+S_e)/2 只取决于 normal↔abnormal 边界。SupCon 让模型更聚焦正常类，但牺牲了异常检测。

**Q：权重融合为什么有效？**
因为两个模型的"偏好"不同：基线偏 S_e，SupCon 偏 S_p。融合后两方面都提升。

**Q：融合 loss 为什么不如权重融合？**
因为融合 loss 在训练时两个 loss 会"打架"，模型只能学到一个方向的特征。权重融合是"事后融合"，保留了多样性。

---

## 致谢

基于 [Patch-Mix Contrastive Learning](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）的代码实现。
