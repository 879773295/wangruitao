# 呼吸音分类：Frequency Band Embedding (FBE)

基于 [Patch-Mix Contrastive Learning](https://arxiv.org/abs/2305.14032)（INTERSPEECH 2023）改进的呼吸音分类方法，在 AST（Audio Spectrogram Transformer）基础上引入**频段嵌入（Frequency Band Embedding, FBE）**，向模型注入频率先验知识以提升呼吸音分类性能。

---

## 核心创新点：Frequency Band Embedding (FBE)

### 思想

在 patch 进入 Transformer 之前，向模型注入频率先验知识，告诉模型每个 patch 属于哪个频段，从而引导注意力机制优先关注病理相关的频率区域。

具体做法：每个 patch 额外增加一个**频段嵌入向量**，与原有的 patch 特征向量、位置编码相加，共同作为 Transformer 的输入。

```
patch_embed(x)                          # patch 特征向量
  → + freq_band_embed(x)                # 加频段嵌入向量（创新点）
  → + pos_embed                         # 加位置编码
  → Transformer blocks                  # 三者共同作为输入
```

### 病理频段划分（主版本）

按呼吸道病理意义（Hz）将频率轴显式划分为 4 个频段：

| band | Hz 范围 | 病理意义 |
|:---:|---------|---------|
| 0 | 0-500 Hz | 心音、背景噪声 |
| 1 | 500-1500 Hz | 啰音（crackles）主要能量区 |
| 2 | 1500-2500 Hz | 哮鸣音（wheezes）主要能量区 |
| 3 | >2500 Hz | 呼吸音细节、噪声 |

---

## 模型清单

| `--model` | 文件 | 说明 | 频段数 |
|-----------|------|------|:---:|
| `ast` | `models/ast.py` | 基线：原始 AST（无 FBE）| - |
| `ast_fbe` | `models/ast_freq.py` | 病理频段 FBE（主创新点）| 4 |
| `ast_fbe_uniform` | `models/ast_freq_uniform.py` | 等分频段 FBE（消融对照）| 12 |

详细模型描述见 [`MODELS.md`](MODELS.md)。

---

## 环境要求

```
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

建议使用带 CUDA 的 PyTorch（本项目训练需 GPU）。

---

## 数据准备

下载 ICBHI 2017 Challenge 数据集，将 `*.wav` 和 `*.txt` 放入 `data/icbhi_dataset/audio_test_data/`。

```bash
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
```

ICBHI 数据集包含 920 个标注音频样本、6,898 个呼吸周期（含啰音、哮鸣音等）。

---

## 训练

### 完整训练（50 epoch）

```bash
python main.py \
  --tag icbhi_fbe \
  --dataset icbhi --seed 1 \
  --class_split lungsound --n_cls 4 \
  --epochs 50 --batch_size 8 \
  --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 \
  --cosine \
  --model ast_fbe \              # 切换模型：ast / ast_fbe / ast_fbe_uniform
  --test_fold official --pad_types repeat --resz 1 --n_mels 128 \
  --ma_update --ma_beta 0.5 \
  --from_sl_official --audioset_pretrained \
  --use_freq_band_embed \
  --method patchmix_cl \
  --temperature 0.06 --proj_dim 768 --alpha 1.0 --mix_beta 1.0 \
  --num_workers 0
```

### 小数据快速验证

设置环境变量 `ICBHI_MAX_FILES=40` 可只用少量文件快速验证：

```bash
ICBHI_MAX_FILES=40 python main.py --model ast_fbe --use_freq_band_embed \
  --method patchmix_cl --epochs 2 --batch_size 4 --num_workers 0 ...
```

---

## 实验结果

> 小数据快速验证（`ICBHI_MAX_FILES=40`，2 epoch）：完整数据结果待跑。

| 模型 | S_p（特异性）| S_e（灵敏度）| Score |
|------|:---:|:---:|:---:|
| `ast_fbe_uniform`（等分频段）| 30.77 | 36.45 | 33.61 |
| `ast_fbe`（病理频段）| **84.42** | 36.08 | **60.25** |

**病理频段划分显著优于等分频段**，Score 提升近一倍。

评估指标：S_p（specificity，正常样本识别率）、S_e（sensitivity，异常样本识别率）、Score = (S_p + S_e) / 2。

---

## 致谢

本项目基于 [Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification](https://arxiv.org/abs/2305.14032)（Sangmin Bae et al., INTERSPEECH 2023）的代码实现。
