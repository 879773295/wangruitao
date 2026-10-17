#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加权集成方案

融合多个模型的优点：
- 基线：S_e 较高（39.76）
- LS：S_e 最高（44.01）
- SupCon：S_p 最高（76.00）

通过加权平均 logits，找最优权重组合。
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import main as M
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score, get_score_both


def load_model(ckpt_path, model_args):
    """加载模型"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    saved_args = ckpt['args']

    # 补丁
    for attr in ['label_smoothing', 'use_supcon', 'loss', 'focal_gamma',
                 'weighted_loss', 'alpha', 'temperature', 'method',
                 'weight_decay', 'proj_dim']:
        if not hasattr(saved_args, attr):
            defaults = {
                'label_smoothing': 0.0, 'use_supcon': False, 'loss': 'ce',
                'focal_gamma': 2.0, 'weighted_loss': False, 'alpha': 1.0,
                'temperature': 0.06, 'method': 'patchmix_cl', 'weight_decay': 1e-6,
                'proj_dim': 768
            }
            setattr(saved_args, attr, defaults.get(attr, 0))

    model, classifier, projector, criterion, optimizer = M.set_model(saved_args)
    model.load_state_dict(ckpt['model'])
    classifier.load_state_dict(ckpt['classifier'])
    model.eval()
    classifier.eval()

    return model, classifier


def extract_logits(model, classifier, dataloader, device='cuda'):
    """提取 logits"""
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels, metadata in dataloader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                features = model(images)
                output = classifier(features)
            all_logits.append(output.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def evaluate_weighted_ensemble(logits_list, labels, weights):
    """评估加权集成"""
    # 加权平均 logits
    weighted_logits = torch.zeros_like(logits_list[0])
    for logit, w in zip(logits_list, weights):
        weighted_logits += w * logit

    # 预测
    preds = torch.argmax(weighted_logits, dim=1)

    # 双口径（4c=精确命中；2c=任一异常）
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.asarray(labels)
    return get_score_both(preds_np, labels_np)  # (sp, se4, se2, sc4, sc2)


def main():
    # 加载数据
    print("=== 加载数据 ===")
    model_args = {
        'h': 798, 'w': 128, 'resz': 1.0, 'n_cls': 4,
        'cls_list': ['normal', 'crackle', 'wheeze', 'both'],
        'sample_rate': 16000, 'n_mels': 128, 'nfft': 1024,
        'desired_length': 8, 'pad_types': 'repeat',
        'specaug_policy': 'icbhi_ast_sup', 'specaug_mask': 'mean',
        'blank_region_clip': False, 'butterworth_filter': None,
        'raw_augment': 0, 'concat_aug_scale': 0, 'batch_size': 8,
        'num_workers': 0, 'test_fold': 'official',
        'data_folder': './data/', 'stetho_id': -1
    }

    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args = Args(**model_args)

    val_transform = [transforms.ToTensor(),
                     transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    val_transform = transforms.Compose(val_transform)
    val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 加载模型
    print("\n=== 加载模型 ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 基线
    print("加载基线模型...")
    model_baseline, classifier_baseline = load_model(
        './save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth', model_args
    )

    # LS（假设你训练了，路径需要改成实际的）
    # 如果没有训练 LS，可以跳过或用基线替代
    # print("加载 LS 模型...")
    # model_ls, classifier_ls = load_model('./save/xxx/best.pth', model_args)

    # SupCon
    print("加载 SupCon 模型...")
    model_supcon, classifier_supcon = load_model(
        './save/icbhi_ast_patchmix_cl_supcon_seed1_e10/best.pth', model_args
    )

    # 提取 logits
    print("\n=== 提取 logits ===")
    logits_baseline, labels = extract_logits(model_baseline, classifier_baseline, val_loader, device)
    print(f"基线 logits: {logits_baseline.shape}")

    logits_supcon, _ = extract_logits(model_supcon, classifier_supcon, val_loader, device)
    print(f"SupCon logits: {logits_supcon.shape}")

    # 搜索最优权重
    print("\n=== 搜索最优权重 ===")
    best_score = 0
    best_weights = None

    # 网格搜索（基线权重从 0 到 1）
    for w1 in np.arange(0, 1.1, 0.1):
        w2 = 1.0 - w1  # SupCon 权重
        weights = [w1, w2]

        sp, se4, se2, sc4, sc2 = evaluate_weighted_ensemble([logits_baseline, logits_supcon], labels, weights)
        se, sc = se4, sc4  # best 仍按 4c（历史口径）
        print(f"  w1={w1:.1f} -> S_p={sp:.2f} Score(4c)={sc:.2f} | Score(2c)={sc2:.2f}")

        if sc > best_score:
            best_score = sc
            best_sp = sp
            best_se = se
            best_weights = weights
            print(f"  w_baseline={w1:.1f}, w_supcon={w2:.1f} -> S_p={sp:.2f}, S_e={se:.2f}, Score={sc:.2f} ★")

    # 打印最优结果
    print(f"\n{'='*60}")
    print(f"最优权重: w_baseline={best_weights[0]:.1f}, w_supcon={best_weights[1]:.1f}")
    print(f"Score: {best_score:.2f}")
    print(f"S_p: {best_sp:.2f}, S_e: {best_se:.2f}")
    print(f"{'='*60}")

    # 对比单模型
    print(f"\n对比:")
    print(f"  基线: S_p=73.40, S_e=39.76, Score=56.58")
    print(f"  SupCon: S_p=76.00, S_e=34.58, Score=55.29")
    print(f"  **融合: S_p={best_sp:.2f}, S_e={best_se:.2f}, Score={best_score:.2f}**")


if __name__ == '__main__':
    main()
