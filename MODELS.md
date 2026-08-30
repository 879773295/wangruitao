# 模型清单（MODELS）

本文档记录项目所有模型版本及其对应描述，用于论文消融实验的对照。

---

## 一、模型总览

| `--model` | 文件 | 创新点 | 频段数 | 状态 |
|-----------|------|--------|:---:|------|
| `ast` | `models/ast.py` | 基线：原始 AST（无 FBE） | - | 原作者实现 |
| `ast_fbe` | `models/ast_freq.py` | 病理频段 FBE | 4 | ✅ 当前主版本 |
| `ast_fbe_patho` | `models/ast_freq.py` | 同上（别名，语义更明确）| 4 | ✅ |
| `ast_fbe_uniform` | `models/ast_freq_uniform.py` | 等分频段 FBE | 12 | ✅ 消融对比 |

---

## 二、模型详细描述

### 1. `ast` — 基线模型

- **来源**：原作者 Patch-Mix Contrastive Learning（INTERSPEECH 2023）
- **结构**：Audio Spectrogram Transformer (AST)，基于 ViT（base384），ImageNet + AudioSet 预训练
- **创新点**：无 FBE，作为消融实验的基线

### 2. `ast_fbe` — 病理频段 FBE（主创新点）

- **创新点**：Frequency Band Embedding (FBE) —— 病理频段划分版本
- **核心思想**：在 patch 进入 Transformer 之前，向模型注入频率先验知识，引导注意力机制优先关注病理相关的频率区域
- **实现**：每个 patch 额外增加一个频段嵌入向量，与 patch 特征向量、位置编码相加，共同作为 Transformer 输入
- **频段划分**：按呼吸道病理意义（Hz）显式划分为 4 个频段

| band | Hz 范围 | 病理意义 |
|:---:|---------|---------|
| 0 | 0-500 Hz | 心音、背景噪声 |
| 1 | 500-1500 Hz | 啰音（crackles）主要能量区 |
| 2 | 1500-2500 Hz | 哮鸣音（wheezes）主要能量区 |
| 3 | >2500 Hz | 呼吸音细节、噪声 |

### 3. `ast_fbe_uniform` — 等分频段 FBE（消融对照）

- **创新点**：Frequency Band Embedding (FBE) —— 等分频段版本
- **与主创新点的区别**：将 128 mel bins 频率轴**等分**为 12 个 patch，每个 patch 对应一个独立频段嵌入（隐式先验，让模型自己学习频段重要性）
- **用途**：作为消融实验，验证"病理频段划分"相比"等分频段"的优势

---

## 三、实验结果

> 注：以下为小数据快速验证（`ICBHI_MAX_FILES=40`，2 epoch，batch_size 4）结果，仅用于验证趋势；完整数据（539 文件 × 50 epoch）结果待跑。

| 模型 | S_p（特异性）| S_e（灵敏度）| Score |
|------|:---:|:---:|:---:|
| `ast`（基线）| - | - | -（待跑）|
| `ast_fbe_uniform`（等分）| 30.77 | 36.45 | 33.61 |
| `ast_fbe`（病理频段）| **84.42** | 36.08 | **60.25** |

**结论**：病理频段划分显著优于等分频段，Score 提升近一倍（33.61 → 60.25）。

---

## 四、训练命令（统一模板）

三个模型只用 `--model` 切换，其余参数一致：

```bash
cd D:/boss/wangruitao/patch-mix_contrastive_learning

# 基线
"F:\anaconda\envs\comfyui\python.exe" main.py --model ast ... --method patchmix_cl --num_workers 0

# 病理频段 FBE
"F:\anaconda\envs\comfyui\python.exe" main.py --model ast_fbe --use_freq_band_embed ... --method patchmix_cl --num_workers 0

# 等分频段 FBE
"F:\anaconda\envs\comfyui\python.exe" main.py --model ast_fbe_uniform --use_freq_band_embed ... --method patchmix_cl --num_workers 0
```

---

## 五、如何新增一个模型版本

1. **新建文件**（不覆盖）：`models/ast_freq_xxx.py`，复制现有版本改创新点
2. **注册**：在 `models/__init__.py` 的 `_backbone_class_map` 加一行
3. **描述**：在本文件（MODELS.md）加一行模型描述
4. **提交**：`git add . && git commit -m "feat: 新增 xxx 模型" && git push`
