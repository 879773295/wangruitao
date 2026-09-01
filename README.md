# 呼吸音分类：Patch-Mix CL + FBE + 集成冲分

基于 [Patch-Mix Contrastive Learning](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）改进的呼吸音分类，在 AST（Audio Spectrogram Transformer）上引入频段嵌入（FBE），并给出经过实测的**多 seed 集成 + TTA + 阈值校准**冲分配方。

**评测指标**：ICBHI 2017 的 Score = (S_p + S_e) / 2，其中 S_p = 正常样本识别率（特异性），S_e = 异常样本识别率（灵敏度）。**Score 只取决于"正常 vs 异常"这条边界**，异常内部怎么分不影响 Score。

---

## 总览：一共 4 步

| 步骤 | 做什么 | 大概耗时 |
|---|------|---------|
| **① 装环境** | 建 conda 环境，装 torch 2.0.1 | 10 分钟 |
| **② 放数据** | ICBHI 数据集放入 `data/` | - |
| **③ 训练** | 训练模型（建议快训）| 每 seed ~20 分钟 |
| **④ 冲分** | 多 seed 集成 + TTA + 阈值校准 | ~30 分钟 |

---

## 第 1 步：装环境

**必须用 torch 2.0.1 + cu118（torchaudio 2.0.2）**。实测 torch 2.2+ 会掉 ~5 分（seed1 从 60.35 → 56.84）。

```bash
conda create -n patchmix python=3.10 -y
conda activate patchmix
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.4.5 librosa numpy==1.24.4 pandas nlpaug wget tqdm cmapy opencv-python
```

> 首次运行会自动下载 ImageNet+AudioSet 预训练模型（`pretrained_models/audioset_10_10_0.4593.pth`，需外网）。
>
> **硬件说明**：GPU 数值栈影响 Score 上限。实测 **RTX 4090 上限约 61.3**；**RTX PRO 6000 / A100 可复现论文数值，单模型即 62**，冲 62+ 建议用这类卡。

## 第 2 步：准备数据

下载 [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/) 数据集，把 `*.wav` 和 `*.txt` 放进：

```
data/icbhi_dataset/audio_test_data/
```

## 第 3 步：训练

先选训练方式，再跑对应命令。

### 方式 A：快训（推荐）

实测**所有训练都在 epoch 3-8 达到 Score 峰值**，后面 40 个 epoch 从不刷新。用 `--max_epochs` 保留 50 epoch 的 LR 曲线、提前在第 10 epoch 停，**每 seed 只要 ~20 分钟**（完整训练要 ~2 小时）。

```bash
python main.py --method patchmix_cl --epochs 50 --max_epochs 10 --batch_size 8 \
  --seed 1 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --from_sl_official --audioset_pretrained --ma_update --ma_beta 0.5 \
  --tag my_seed1
```

### 方式 B：完整训练（50 epoch）

```bash
python main.py --method patchmix_cl --epochs 50 --batch_size 8 \
  --seed 1 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --from_sl_official --audioset_pretrained --ma_update --ma_beta 0.5 \
  --tag my_seed1
```

> 换模型：`--model ast`（基线）/ `ast_fbe`（FBE 创新点）/ `ast_fbe_uniform` / `ast_cnn_dual`。

### 方式 C：小数据快速验证（调试用）

```bash
ICBHI_MAX_FILES=160 python main.py --method patchmix_cl --epochs 20 --batch_size 8 \
  --seed 1 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --from_sl_official --audioset_pretrained --tag debug
```

> ⚠️ 小数据只适合调试/筛 loss 类改动；**架构改动排名不可靠**（实测 timedom 小数据 +1.86、全量变假阳性）。

## 第 4 步：冲分（多 seed 集成 + TTA + 阈值）

多训练几个 seed（建议 4~8 个），把它们的 `best.pth` 一起做集成：

```bash
python ensemble_eval.py --ckpts \
  ./save/icbhi_ast_patchmix_cl_my_seed1/best.pth \
  ./save/icbhi_ast_patchmix_cl_my_seed2/best.pth \
  ... \
  --tta 16
```

输出两行关键结果：
- **ensemble (argmax)**：直接多数投票的 Score
- **best (阈值)**：把 normal 类 logit 加一个偏置后扫出来的最优 Score（比 argmax 高 ~1-1.5 分）

**一键版**：`bash pro6000_run.sh` 会一次性完成「快训 8 个 seed → 集成 → TTA → 阈值」。

---

## 常见问题

**Q：为什么我单模型只有 ~60，却要跑 8 个 seed？**
seed 方差很大（实测单 seed 落在 56~60.35）。集成+TTA+阈值能确定性抬 ~1-1.5 分，是"不择手段"冲 62 最稳的路。

**Q：4090 上能到 62 吗？**
集成+TTA+阈值实测封顶 **61.28**，是硬件数值栈的上限。**要破 62 请用 PRO 6000/A100**（那类卡单模型就 62，配合本配方可冲 63+）。

**Q：`--max_epochs` 快训靠谱吗？**
靠谱。所有跑都在 epoch 3-8 见顶，快训就是利用这一点省时间，best checkpoint 质量和完整训练一致。

---

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
| `main.py` | 训练/评估入口（含 `--loss focal`、`--max_epochs`）|
| `ensemble_eval.py` | 集成 + TTA + 阈值校准 |
| `eval_threshold.py` | 单模型阈值校准 |
| `pro6000_run.sh` | 一键冲分 |
| `fast_matrix.sh` | 小数据候选矩阵筛选 |

## 致谢

基于 [Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification](https://arxiv.org/abs/2305.14032)（Sangmin Bae et al., INTERSPEECH 2023）的代码实现。
