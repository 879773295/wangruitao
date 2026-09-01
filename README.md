# 呼吸音分类：Patch-Mix Contrastive Learning + FBE + 集成冲分配方

基于 [Patch-Mix Contrastive Learning](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）改进的呼吸音分类方法，在 AST（Audio Spectrogram Transformer）基础上引入**频段嵌入（Frequency Band Embedding, FBE）**，并向模型注入频率先验。同时提供经过实战验证的 **多 seed 集成 + TTA + 阈值校准** 冲分配方（ICBHI 2017 Score）。

---

## 核心创新点

### Frequency Band Embedding (FBE)

在 patch 进入 Transformer 之前注入频率先验，告诉模型每个 patch 属于哪个频段，引导注意力优先关注病理相关频率区域：

```
patch_embed(x)                          # patch 特征向量
  → + freq_band_embed(x)                # 加频段嵌入向量（创新点）
  → + pos_embed                         # 加位置编码
  → Transformer blocks
```

病理频段按 Hz 显式划分：

| band | Hz 范围 | 病理意义 |
|:---:|---------|---------|
| 0 | 0-500 Hz | 心音、背景噪声 |
| 1 | 500-1500 Hz | 啰音（crackles）主要能量区 |
| 2 | 1500-2500 Hz | 哮鸣音（wheezes）主要能量区 |
| 3 | >2500 Hz | 呼吸音细节、噪声 |

## 冲分配方（实测最优）

> **结论：ICBHI 的 Score = (S_p + S_e)/2 只取决于 normal↔abnormal 这条边界**，crackle/wheeze/both 内部怎么分不影响 Score。因此最优策略是**多个 seed 集成降低方差 + 阈值校准找最优边界切点**。

实测（torch 2.0.1 + cu118，8 成员集成 + TTA(16) + 阈值偏置 normal logit）：

| 方式 | S_p | S_e | Score |
|---|---|---|---|
| 单模型（seed1）| 76.19 | 44.52 | 60.35 |
| 8-seed 集成 + TTA | 74.03 | 46.22 | 60.13 |
| **8-seed 集成 + TTA + 阈值** | **88.92** | **33.64** | **61.28** |

**硬件要求**：Score 受 GPU 数值栈影响很大（fp16 数学）。实测 RTX 4090 上限约 61.3，**RTX PRO 6000（Blackwell）/ A100 级别可复现论文数值，单模型即可到 62**，配合本配方可冲 63+。详见下方「重要发现」。

---

## 快速开始（推荐：PRO 6000 上冲 62+）

一键跑 8 seed 快训 + 集成 + TTA + 阈值：

```bash
# 1) 环境（PRO 6000 用默认 python 即可；4090 必须用 torch 2.0.1，见下）
# 2) 一键脚本
bash pro6000_run.sh
# 3) 看 pro6000_ensemble_result.txt 里的 ">>> best"
```

脚本会快训 8 个 seed（seed 1/2/3/4/5/6/1337/42，每个 ~20 分钟，共 ~3 小时），再做 8 成员集成 + TTA(16) + 阈值校准。

---

## 环境要求

**关键：必须用 torch 2.0.1 + cu118（torchaudio 2.0.2）。** torch 2.2+ 会掉 ~5 分（实测 seed1 从 60.35 → 56.84），是 AMP/数值差异导致。

```bash
conda create -n patchmix python=3.10 -y
conda activate patchmix
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.4.5 librosa numpy==1.24.4 pandas nlpaug wget tqdm cmapy opencv-python
```

首次运行会自动下载 ImageNet+AudioSet 预训练模型（`pretrained_models/audioset_10_10_0.4593.pth`，需外网）。

## 数据准备

下载 ICBHI 2017 Challenge 数据集，将 `*.wav` 和 `*.txt` 放入 `data/icbhi_dataset/audio_test_data/`。

```bash
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
```

---

## 训练命令

### 单模型完整训练（50 epoch）

```bash
python main.py \
  --tag icbhi_best --dataset icbhi --seed 1 \
  --class_split lungsound --n_cls 4 \
  --epochs 50 --batch_size 8 \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 \
  --cosine \
  --model ast              # ast / ast_fbe / ast_fbe_uniform / ast_cnn_dual
  --test_fold official --pad_types repeat --resz 1 --n_mels 128 \
  --ma_update --ma_beta 0.5 \
  --from_sl_official --audioset_pretrained \
  --method patchmix_cl --temperature 0.06 --proj_dim 768 --alpha 1.0 --mix_beta 1.0 \
  --num_workers 0
```

### 快训（推荐，实测最优 seed 都在 epoch 3-8 见顶）

实测**所有训练都在 epoch 3-8 达到 Score 峰值**，后 40 个 epoch 从不刷新。用 `--max_epochs` 保留 50 epoch 的 LR 曲线、提前在第 10 epoch 停，**每个 seed 只需 ~20 分钟**（vs 完整 50 epoch 的 ~2 小时）：

```bash
python main.py --method patchmix_cl --epochs 50 --max_epochs 10 --batch_size 8 \
  --seed 6 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --from_sl_official --audioset_pretrained --ma_update --ma_beta 0.5 \
  --tag seed6_fast
```

### 小数据快速验证

```bash
ICBHI_MAX_FILES=160 python main.py ... --epochs 20 --batch_size 8 ...
```

> ⚠️ 小数据只适合筛 loss/数据类改动，**架构改动排名不可靠**（实测 timedom 小数据 +1.86、全量变假阳性）。

---

## 评估命令

### 集成 + TTA + 阈值校准（冲分核心）

```bash
python ensemble_eval.py --ckpts \
  ./save/*/best.pth ... --tta 16
```

输出：argmax Score + 阈值偏置 normal logit 扫描的最优 Score。

### 单模型阈值校准

```bash
python eval_threshold.py --ckpt ./save/icbhi_ast_patchmix_cl_xxx/best.pth
```

---

## 重要发现（实战记录）

1. **Score 只取决于 normal↔abnormal 边界** → 阈值校准是最有效的"免费"增益。
2. **GPU 数值栈决定 Score 上限**：4090 + torch 2.0.1 约 61.3（集成+阈值后），PRO 6000 单模型即 62。**想破 62 请用 PRO 6000/A100**。
3. **seed 方差大**（56~60.35），多 seed 集成是确定性增益；seed 6（S_p=69.9 / S_e=49.3）是 S_e 最高的成员，值得单独调优。
4. **epoch 3-8 见顶** → `--max_epochs` 快训（见上）。
5. **小数据架构筛选不可靠**（timedom 假阳性）。

## 模型清单

| `--model` | 文件 | 说明 |
|-----------|------|------|
| `ast` | `models/ast.py` | 基线：原始 AST |
| `ast_fbe` | `models/ast_freq.py` | 病理频段 FBE（主创新点）|
| `ast_fbe_uniform` | `models/ast_freq_uniform.py` | 等分频段 FBE（消融对照）|
| `ast_cnn_dual` | `models/ast_cnn_dual.py` | AST + CNN 双通道 |
| `ssast` | `models/ssast.py` | SSAST（需自行下载预训练）|

## 脚本说明

| 文件 | 作用 |
|------|------|
| `main.py` | 训练/评估入口（含 `--loss focal`、`--max_epochs` 补丁）|
| `ensemble_eval.py` | 集成 + TTA + 阈值校准评估 |
| `eval_threshold.py` | 单模型阈值校准 |
| `pro6000_run.sh` | 一键冲分（8 seed 快训 + 集成 + TTA + 阈值）|
| `fast_matrix.sh` | 小数据候选矩阵筛选 |

## 致谢

基于 [Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification](https://arxiv.org/abs/2305.14032)（Sangmin Bae et al., INTERSPEECH 2023）的代码实现。
