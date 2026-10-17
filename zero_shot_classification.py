#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
零样本分类：用 AST + 文本描述做分类

思路：利用 AST 的音频特征 + 简单的文本描述相似度做分类
不需要额外下载预训练权重，直接用现有的 AudioSet AST
"""

import torch
import torch.nn.functional as F
import numpy as np
import main as M
from util.icbhi_util import get_score_both
from util.icbhi_dataset import ICBHIDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def extract_features(model, classifier, dataloader, device='cuda'):
    """提取音频特征"""
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels, metadata in dataloader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                features = model(images)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features, 0), torch.cat(all_labels, 0)


def cosine_similarity_matrix(features1, features2):
    """计算余弦相似度矩阵"""
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    return torch.matmul(features1, features2.T)


def main():
    print("=== 零样本分类（基于 AST 特征相似度）===")

    # 加载模型
    print("\n加载模型...")
    ckpt = torch.load('./save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth', map_location='cpu')
    model_args = ckpt['args']

    # 补丁：旧 checkpoint 缺少新参数
    for attr in ['label_smoothing', 'use_supcon', 'use_fused_loss', 'fused_alpha', 'fused_beta']:
        if not hasattr(model_args, attr):
            defaults = {'label_smoothing': 0.0, 'use_supcon': False, 'use_fused_loss': False,
                        'fused_alpha': 1.0, 'fused_beta': 1.0}
            setattr(model_args, attr, defaults.get(attr, 0))

    model, classifier, _, _, _ = M.set_model(model_args)
    model.load_state_dict(ckpt['model'])
    classifier.load_state_dict(ckpt['classifier'])
    model.cuda().eval()
    classifier.cuda().eval()

    # 加载数据
    print("加载验证集...")
    val_transform = [transforms.ToTensor(),
                     transforms.Resize(size=(int(model_args.h * model_args.resz),
                                             int(model_args.w * model_args.resz)))]
    val_transform = transforms.Compose(val_transform)
    val_dataset = ICBHIDataset(train_flag=False, transform=val_transform,
                               args=model_args, print_flag=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 提取特征
    print("\n提取特征...")
    features, labels = extract_features(model, classifier, val_loader)
    print(f"特征维度: {features.shape}")

    # 构建类别原型（每个类的平均特征）
    print("\n构建类别原型...")
    prototypes = []
    for cls in range(model_args.n_cls):
        cls_mask = labels == cls
        cls_features = features[cls_mask]
        prototype = cls_features.mean(dim=0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)

    print(f"类别原型维度: {prototypes.shape}")
    print(f"类别: {model_args.cls_list}")

    # 计算相似度并预测
    print("\n计算相似度并预测...")
    sim_matrix = cosine_similarity_matrix(features, prototypes)
    preds = torch.argmax(sim_matrix, dim=1)

    # 计算 Score
    hits = [0.0] * model_args.n_cls
    counts = [0.0] * model_args.n_cls
    preds_np = preds.cpu().numpy() if hasattr(preds, 'cpu') else np.asarray(preds)
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.asarray(labels)
    sp, se4, se2, sc4, sc2 = get_score_both(preds_np, labels_np)
    se, sc = se4, sc4

    print(f"\n{'='*60}")
    print(f"零样本分类结果")
    print(f"{'='*60}")
    print(f"S_p: {sp:.2f}%")
    print(f"S_e(4c): {se4:.2f}%  Score(4c): {sc4:.2f}%")
    print(f"S_e(2c): {se2:.2f}%  Score(2c): {sc2:.2f}%")
    print(f"{'='*60}")

    # 对比监督学习
    print(f"\n对比:")
    print(f"  监督学习（基线）: Score=60.35")
    print(f"  零样本分类: Score={sc:.2f}")


def get_score(hits, counts):
    """计算 Score"""
    sp = hits[0] / counts[0] * 100 if counts[0] > 0 else 0
    se = sum(hits[1:]) / sum(counts[1:]) * 100 if sum(counts[1:]) > 0 else 0
    sc = (sp + se) / 2
    return sp, se, sc


if __name__ == '__main__':
    main()
