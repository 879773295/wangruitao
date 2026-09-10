"""
SAM + Band-aware Patch-Mix 训练脚本（支持消融）

创新点（可独立开关）：
1. Same-band Patch-Mix：只在相同频段内交换 patch
2. Band Importance Estimator：学习频段重要性权重
3. Frequency-weighted λ：按频段重要性计算混合比例
4. BandCL Loss：约束混合后频段与源样本一致

输出：S_p, S_e, Score
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix

from src.dataset import ASTDataset
from src.model_bandaware import build_model
from src.sam import SAM


def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 数据加载
    print(f"Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件未找到: {args.data_path}. 请先运行 preprocess.py")

    data = np.load(args.data_path)
    X_train, y_train, d_train = data['X_train'], data['y_train'], data['device_train']
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    counts = np.bincount(y_train)
    weights = [1.0/counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True),
        batch_size=args.batch_size,
        sampler=sampler
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False
    )

    # 模型和优化器
    print("Preparing model...")
    print(f"  use_same_band_mix: {args.use_same_band_mix}")
    print(f"  use_band_importance: {args.use_band_importance}")
    print(f"  use_freq_weighted_lam: {args.use_freq_weighted_lam}")
    print(f"  use_bandcl: {args.use_bandcl}")

    model = build_model(
        num_classes=4,
        num_bands=args.num_bands,
        use_same_band_mix=args.use_same_band_mix,
        use_band_importance=args.use_band_importance,
        use_freq_weighted_lam=args.use_freq_weighted_lam,
        use_bandcl=args.use_bandcl
    ).to(DEVICE)

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练循环
    print("Train begins")
    best_score = 0.0
    best_sp = 0.0
    best_se = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播（带创新点）
            logits, band_features_src, band_features_mix, band_importance, index = model(
                inputs, patch_mix=True, mix_lam=0.5
            )

            # 分类损失
            loss = criterion(logits, labels)

            # 创新点 4：BandCL Loss（可选）
            if args.use_bandcl:
                bandcl_loss = model.compute_bandcl_loss(band_features_src, band_features_mix)
                loss += args.bandcl_weight * bandcl_loss

            # SAM 两步更新
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # 第二步：重新计算 loss
            logits2, _, _, _, _ = model(inputs, patch_mix=True, mix_lam=0.5)
            loss2 = criterion(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        # 验证
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(inputs)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        cm = confusion_matrix(all_labels, all_preds)
        se = np.sum(cm[1:, 1:]) / np.sum(cm[1:, :]) if np.sum(cm[1:, :]) > 0 else 0
        sp = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0
        score = (se + sp) / 2

        # 计算平均 loss
        avg_loss = running_loss / len(train_loader)

        # 打印结果
        print(f'\nEpoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}')
        print(f' * S_p: {sp*100:.2f}, S_e: {se*100:.2f}, Score: {score*100:.2f}')

        # 保存最优模型
        if score > best_score:
            best_score = score
            best_sp = sp
            best_se = se
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f'   * New best: S_p: {sp*100:.2f}, S_e: {se*100:.2f}, Score: {score*100:.2f}')

    print(f'\n * Best Score: {best_score*100:.2f} (S_p: {best_sp*100:.2f}, S_e: {best_se*100:.2f})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM + Band-aware Patch-Mix 训练（支持消融）")
    parser.add_argument("--data_path", type=str, default="./icbhi_ast_16k_8s_metadata.npz", help="预处理后的数据文件路径")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--num_bands", type=int, default=4, help="频段数量")

    # 消融实验参数
    parser.add_argument("--use_same_band_mix", action="store_true", default=True, help="启用 Same-band Patch-Mix")
    parser.add_argument("--no_same_band_mix", action="store_false", dest="use_same_band_mix", help="禁用 Same-band Patch-Mix")
    parser.add_argument("--use_band_importance", action="store_true", default=True, help="启用 Band Importance Estimator")
    parser.add_argument("--no_band_importance", action="store_false", dest="use_band_importance", help="禁用 Band Importance Estimator")
    parser.add_argument("--use_freq_weighted_lam", action="store_true", default=True, help="启用 Frequency-weighted λ")
    parser.add_argument("--no_freq_weighted_lam", action="store_false", dest="use_freq_weighted_lam", help="禁用 Frequency-weighted λ")
    parser.add_argument("--use_bandcl", action="store_true", default=True, help="启用 BandCL Loss")
    parser.add_argument("--no_bandcl", action="store_false", dest="use_bandcl", help="禁用 BandCL Loss")
    parser.add_argument("--bandcl_weight", type=float, default=1.0, help="BandCL Loss 权重")

    args = parser.parse_args()
    train(args)
