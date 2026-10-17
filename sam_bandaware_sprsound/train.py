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
from src.dataset_npz import NPZDataset
from src.model_bandaware import build_model
from src.sam import SAM


def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 数据加载
    print(f"Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件未找到: {args.data_path}. 请先运行 preprocess_sprsound.py")

    data = np.load(args.data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    counts = np.bincount(y_train)
    weights = [1.0/counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    train_loader = DataLoader(
        NPZDataset(X_train, y_train, processor, augment=True),
        batch_size=args.batch_size,
        sampler=sampler
    )
    test_loader = DataLoader(
        NPZDataset(X_test, y_test, processor, augment=False),
        batch_size=args.batch_size,
        shuffle=False
    )

    # 模型和优化器
    print("Preparing model...")
    print(f"  use_same_band_mix: {args.use_same_band_mix}")
    print(f"  use_band_importance: {args.use_band_importance}")
    print(f"  use_freq_weighted_lam: {args.use_freq_weighted_lam}")
    print(f"  use_bandcl: {args.use_bandcl}")
    print(f"  mix_lam: {args.mix_lam}")
    print(f"  bandcl_weight: {args.bandcl_weight}")
    print(f"  bandcl_loss_type: {args.bandcl_loss_type}")
    print(f"  band_importance_temperature: {args.band_importance_temperature}")

    model = build_model(
        num_classes=args.num_classes,
        num_bands=args.num_bands,
        use_same_band_mix=args.use_same_band_mix,
        use_band_importance=args.use_band_importance,
        use_freq_weighted_lam=args.use_freq_weighted_lam,
        use_bandcl=args.use_bandcl,
        band_importance_hidden_ratio=args.band_importance_hidden_ratio,
        band_importance_temperature=args.band_importance_temperature,
        bandcl_loss_type=args.bandcl_loss_type
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

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            if args.use_patch_mix:
                # PatchMix 模式（带创新点）
                logits, band_features_src, band_features_mix, band_importance, index = model(
                    inputs, patch_mix=True, mix_lam=args.mix_lam
                )

                # 分类损失
                loss = criterion(logits, labels)

                # 创新点 4：BandCL Loss（可选）
                if args.use_bandcl:
                    bandcl_loss = model.compute_bandcl_loss(band_features_src, band_features_mix)
                    loss += args.bandcl_weight * bandcl_loss
            else:
                # 原始 SAM 模式（无 PatchMix）
                outputs = model(inputs, patch_mix=False)
                logits = outputs[0]
                loss = criterion(logits, labels)

            # SAM 两步更新
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # 第二步：重新计算 loss
            if args.use_patch_mix:
                logits2, _, _, _, _ = model(inputs, patch_mix=True, mix_lam=args.mix_lam)
            else:
                logits2 = model(inputs, patch_mix=False)[0]
            loss2 = criterion(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        # 验证
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                logits = outputs[0]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标（使用 QLung 评估方式）
        cm = confusion_matrix(all_labels, all_preds)
        # hits[i] = 第 i 类正确预测的数量
        hits = np.diag(cm)
        counts = np.sum(cm, axis=1)
        # S_p = Normal 类准确率
        sp = hits[0] / (counts[0] + 1e-10)
        # S_e(4c) = 异常类精确命中率（历史口径，best 沿用）
        se = np.sum(hits[1:]) / (np.sum(counts[1:]) + 1e-10)
        score = (se + sp) / 2
        # 2-class 折叠口径（判成任一异常即命中）并列输出
        se2 = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-10)
        score2 = (se2 + sp) / 2

        # 计算平均 loss
        avg_loss = running_loss / len(train_loader)

        # 打印结果
        print(f'\nEpoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}')
        print(f' * S_p: {sp*100:.2f}, S_e(4c): {se*100:.2f}, Score(4c): {score*100:.2f} | S_e(2c): {se2*100:.2f}, Score(2c): {score2*100:.2f}')

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
    parser.add_argument("--data_path", type=str, default="./sprsound_data.npz", help="预处理后的数据文件路径")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--num_bands", type=int, default=4, help="频段数量")
    parser.add_argument("--num_classes", type=int, default=5, help="类别数量（ICBHI=4，SPRSound=5）")

    # 消融实验参数
    parser.add_argument("--use_patch_mix", action="store_true", default=True, help="启用 Patch-Mix（基础数据增强）")
    parser.add_argument("--no_patch_mix", action="store_false", dest="use_patch_mix", help="禁用 Patch-Mix（原始 SAM）")
    parser.add_argument("--use_same_band_mix", action="store_true", default=True, help="启用 Same-band Patch-Mix")
    parser.add_argument("--no_same_band_mix", action="store_false", dest="use_same_band_mix", help="禁用 Same-band Patch-Mix")
    parser.add_argument("--use_band_importance", action="store_true", default=True, help="启用 Band Importance Estimator")
    parser.add_argument("--no_band_importance", action="store_false", dest="use_band_importance", help="禁用 Band Importance Estimator")
    parser.add_argument("--use_freq_weighted_lam", action="store_true", default=True, help="启用 Frequency-weighted λ")
    parser.add_argument("--no_freq_weighted_lam", action="store_false", dest="use_freq_weighted_lam", help="禁用 Frequency-weighted λ")
    parser.add_argument("--use_bandcl", action="store_true", default=True, help="启用 BandCL Loss")
    parser.add_argument("--no_bandcl", action="store_false", dest="use_bandcl", help="禁用 BandCL Loss")
    parser.add_argument("--bandcl_weight", type=float, default=1.0, help="BandCL Loss 权重")
    parser.add_argument("--bandcl_loss_type", type=str, default="mse", choices=["mse", "l1", "cosine"], help="BandCL Loss 类型")

    # 新增调参参数
    parser.add_argument("--mix_lam", type=float, default=0.5, help="Patch-Mix 基础混合比例")
    parser.add_argument("--band_importance_hidden_ratio", type=int, default=4, help="Band Importance MLP 隐藏层比例")
    parser.add_argument("--band_importance_temperature", type=float, default=1.0, help="Band Importance softmax 温度")

    args = parser.parse_args()
    train(args)
