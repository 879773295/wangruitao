"""
原始 SAM + CustomAST（Mean Pooling + Dropout）在 SPRSound 事件级 7 分类上的训练脚本

- 模型：HuggingFace AST (MIT/ast-finetuned-audioset-10-10-0.4593) + Dropout(0.3) + Linear(768, 7)
- 优化器：SAM（Sharpness-Aware Minimization），两步更新
- 损失：CrossEntropy(label_smoothing=0.1)
- 无 PatchMix（原始 SAM）

评估：
    S_p = Normal（类 0）准确率
    S_e = 异常类（类 1-6）正确预测占比（宏平均）
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
    def __init__(self, num_classes=7):
        super().__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        outputs = self.ast(x)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits


# ---------------- 数据集 ----------------
class ASTDataset(Dataset):
    def __init__(self, X, y, processor, train=True):
        self.X = X
        self.y = y
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        wav = self.X[idx]

        if self.train:
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
        sampler=sampler
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本")
    print(f"训练集标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    model = CustomAST(num_classes=args.num_classes).to(DEVICE)

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("Train begins")
    best_score = 0.0
    best_sp = 0.0
    best_se = 0.0
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            criterion(model(inputs), labels).backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 验证
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
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

        avg_loss = running_loss / len(train_loader)
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

    args = parser.parse_args()
    train(args)
