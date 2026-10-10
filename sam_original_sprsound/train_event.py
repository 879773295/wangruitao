"""
SPRSound 事件级 7 分类 — SAM + Band-aware Patch-Mix（四模块消融版）

移植自 sam_bandaware_sprsound/train.py（5 类消融），模型底座同为 HuggingFace AST：
- 模型：AST (MIT/ast-finetuned-audioset-10-10-0.4593) + Dropout(0.3) + Linear(768, 7)
- 优化器：SAM（两步更新）  损失：CE(label_smoothing=0.1) + bandcl_weight * BandCL

四模块消融开关（默认全关 = 原始 SAM 基线，行为与旧版 train_event.py 完全一致）：
    --use_patch_mix         启用基础 Patch-Mix（随机交换 patch，Exp1）
    --use_same_band_mix     [A] Same-band Patch-Mix（频段内交换）
    --use_band_importance   [B] Band Importance Estimator
    --use_freq_weighted_lam [C] Frequency-weighted λ
    --use_bandcl            [D] BandCL Loss（配合 --bandcl_weight）
    一键消融：bash run_ablation.sh（6 组，与 5 类版实验设计一致）

冒烟测试：
    --max_steps 5       每 epoch 只训 5 个 batch
    --test_limit 24     只在前 24 条测试样本上评估

评估：
    S_p = Normal（类 0）准确率
    S_e = 异常类（类 1-6）正确预测占比
    Score = (S_p + S_e) / 2
    另报告 7 类总体 accuracy 与各类准确率
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import numpy as np
from transformers import ASTFeatureExtractor, ASTModel
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix

from model_bandaware import build_model

# 7 类标签名（与 preprocess_event.py 一致）
CLASS_NAMES = ['Normal', 'Rhonchi', 'Wheeze', 'Stridor',
               'Coarse Crackle', 'Fine Crackle', 'Wheeze+Crackle']


def per_class_accuracy(cm):
    """返回每个类的准确率列表（按类别 0..N-1）"""
    per_cls = cm.diagonal().astype(float) / np.maximum(cm.sum(axis=1), 1)
    return per_cls


# ---------------- SAM 优化器 ----------------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm


# ---------------- 模型 ----------------
class CustomAST(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        outputs = self.ast(x)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits


# ---------------- 数据集 ----------------
class ASTDataset(Dataset):
    def __init__(self, X, y, processor, train=True, aug=True):
        self.X = X
        self.y = y
        self.processor = processor
        self.train = train
        self.aug = aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        wav = self.X[idx]

        if self.train and self.aug:
            if np.random.random() < 0.5:
                wav = wav * np.random.uniform(0.9, 1.1)
            if np.random.random() < 0.5:
                wav = wav + np.random.normal(0, 0.0001, wav.shape)

        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        return input_values, torch.tensor(self.y[idx], dtype=torch.long)


def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    # 消融开关汇总（与 5 类版 train.py 打印一致，便于日志区分组别）
    print(f"  use_patch_mix: {args.use_patch_mix}")
    print(f"  use_same_band_mix: {args.use_same_band_mix}")
    print(f"  use_band_importance: {args.use_band_importance}")
    print(f"  use_freq_weighted_lam: {args.use_freq_weighted_lam}")
    print(f"  use_bandcl: {args.use_bandcl}")
    print(f"  mix_lam: {args.mix_lam}")
    print(f"  bandcl_weight: {args.bandcl_weight}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件未找到: {args.data_path}. 请先运行 preprocess_event.py")

    data = np.load(args.data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    counts = np.bincount(y_train)
    weights = [1.0 / counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    train_loader = DataLoader(
        ASTDataset(X_train, y_train, processor, train=True),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本")
    print(f"训练集标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    model = build_model(
        num_classes=args.num_classes,
        num_bands=args.num_bands,
        use_same_band_mix=args.use_same_band_mix,
        use_band_importance=args.use_band_importance,
        use_freq_weighted_lam=args.use_freq_weighted_lam,
        use_bandcl=args.use_bandcl,
        band_importance_hidden_ratio=args.band_importance_hidden_ratio,
        band_importance_temperature=args.band_importance_temperature,
        bandcl_loss_type=args.bandcl_loss_type,
    ).to(DEVICE)

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("Train begins")
    best_score = 0.0
    best_sp = 0.0
    best_se = 0.0
    best_acc = 0.0
    best_cm = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_run = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for step, (inputs, labels) in enumerate(progress_bar):
            if args.max_steps and step >= args.max_steps:
                break
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            n_run += 1

            # 第一步前向（PatchMix 模式按需开关，与 5 类版 train.py 一致）
            if args.use_patch_mix:
                logits, band_features_src, band_features_mix, band_importance, index = model(
                    inputs, patch_mix=True, mix_lam=args.mix_lam
                )
                loss = criterion(logits, labels)
                if args.use_bandcl:
                    bandcl_loss = model.compute_bandcl_loss(band_features_src, band_features_mix)
                    loss = loss + args.bandcl_weight * bandcl_loss
            else:
                logits = model(inputs, patch_mix=False)[0]
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM 第二步：重新前向（与 5 类版一致，第二步只用 CE）
            if args.use_patch_mix:
                logits2 = model(inputs, patch_mix=True, mix_lam=args.mix_lam)[0]
            else:
                logits2 = model(inputs, patch_mix=False)[0]
            criterion(logits2, labels).backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 验证（eval 一律不混 patch，与 5 类版一致）
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for step, (inputs, labels) in enumerate(test_loader):
                if args.test_limit and step * args.batch_size >= args.test_limit:
                    break
                inputs = inputs.to(DEVICE)
                logits = model(inputs, patch_mix=False)[0]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        # S_p = Normal（类 0）准确率
        sp = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0
        # S_e = 异常类（类 1-6）正确预测占比
        se = np.sum(cm[1:, 1:]) / np.sum(cm[1:, :]) if np.sum(cm[1:, :]) > 0 else 0
        score = (se + sp) / 2
        # 7 类总体 accuracy
        acc = np.sum(np.diag(cm)) / np.sum(cm)
        # 各类准确率
        pca = per_class_accuracy(cm)

        avg_loss = running_loss / max(n_run, 1)
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}')
        print(f' * S_p: {sp*100:.2f}, S_e: {se*100:.2f}, Score: {score*100:.2f}, Acc: {acc*100:.2f}')
        pca_str = '  '.join(f'{CLASS_NAMES[i]}={pca[i]*100:.1f}' for i in range(len(pca)))
        print(f' * 各类准确率: {pca_str}')

        if score > best_score:
            best_score = score
            best_sp = sp
            best_se = se
            best_acc = acc
            best_cm = cm
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f'   * New best: S_p: {sp*100:.2f}, S_e: {se*100:.2f}, Score: {score*100:.2f}')

    print(f'\n * Best Score: {best_score*100:.2f} (S_p: {best_sp*100:.2f}, S_e: {best_se*100:.2f}, Acc: {best_acc*100:.2f})')
    if best_cm is not None:
        best_pca = per_class_accuracy(best_cm)
        best_pca_str = '  '.join(f'{CLASS_NAMES[i]}={best_pca[i]*100:.1f}' for i in range(len(best_pca)))
        print(f' * Best 时各类准确率: {best_pca_str}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原始 SAM 在 SPRSound 事件级 7 分类上训练")
    parser.add_argument("--data_path", type=str, default="./sprsound_event_7class.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_classes", type=int, default=7)

    # ---- 四模块消融（默认全关 = 原始 SAM 基线，与旧版行为一致） ----
    parser.add_argument("--use_patch_mix", action="store_true", default=False,
                        help="启用基础 Patch-Mix（随机交换 patch）")
    parser.add_argument("--use_same_band_mix", action="store_true", default=False,
                        help="[A] Same-band Patch-Mix：只在同频段内交换")
    parser.add_argument("--use_band_importance", action="store_true", default=False,
                        help="[B] Band Importance Estimator：频段重要性权重")
    parser.add_argument("--use_freq_weighted_lam", action="store_true", default=False,
                        help="[C] Frequency-weighted λ：按重要性加权混合比例")
    parser.add_argument("--use_bandcl", action="store_true", default=False,
                        help="[D] BandCL Loss：混合前后频段特征一致性约束")
    parser.add_argument("--num_bands", type=int, default=4, help="频段数量")
    parser.add_argument("--mix_lam", type=float, default=0.5, help="Patch-Mix 基础混合比例")
    parser.add_argument("--bandcl_weight", type=float, default=0.01, help="BandCL Loss 权重")
    parser.add_argument("--bandcl_loss_type", type=str, default="mse",
                        choices=["mse", "l1", "cosine"], help="BandCL Loss 类型")
    parser.add_argument("--band_importance_hidden_ratio", type=int, default=4,
                        help="Band Importance MLP 隐藏层比例")
    parser.add_argument("--band_importance_temperature", type=float, default=1.0,
                        help="Band Importance softmax 温度")

    # ---- 冒烟测试（验证代码能跑通，不出正式结果） ----
    parser.add_argument("--max_steps", type=int, default=0,
                        help=">0 时每个 epoch 只训前 N 个 batch（0 = 全部）")
    parser.add_argument("--test_limit", type=int, default=0,
                        help=">0 时只用前 N 条测试样本评估（0 = 全部）")

    args = parser.parse_args()
    train(args)
