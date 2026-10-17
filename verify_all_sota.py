#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合验证所有 SOTA 模块

在难样本小数据集上快速验证：
1. 基线（PatchMixCL）
2. Attention Pooling（新模型）
3. Label Smoothing
4. SupCon
5. SupCon + Label Smoothing
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from copy import deepcopy

import main as M
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score, get_score_both


def load_model_and_data(ckpt_path):
    """加载模型和数据"""
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

    return saved_args


def build_datasets(args, hard_indices_path='hard_train_indices.json'):
    """构建数据集"""
    with open(hard_indices_path, 'r') as f:
        hard_samples = json.load(f)
    hard_indices = [s['index'] for s in hard_samples]

    # 训练集
    train_transform = [transforms.ToTensor(),
                       transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    train_transform = transforms.Compose(train_transform)
    full_train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=False)
    hard_train_dataset = Subset(full_train_dataset, hard_indices)

    # 验证集
    val_transform = [transforms.ToTensor(),
                     transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    val_transform = transforms.Compose(val_transform)
    val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=False)

    return hard_train_dataset, val_dataset, args


def train_and_evaluate(model_args, experiment_name, model=None, **kwargs):
    """训练并评估"""
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    for k, v in kwargs.items():
        if v is not None:
            print(f"  {k}: {v}")
    print(f"{'='*60}")

    # 构建数据集
    hard_train_dataset, val_dataset, args = build_datasets(model_args)

    # 合并参数
    for k, v in kwargs.items():
        setattr(args, k, v)

    # 设置模型
    if model is None:
        model_obj, classifier, projector, criterion, optimizer = M.set_model(args)
    else:
        model_obj = model
        classifier = deepcopy(model_obj.mlp_head)
        projector = M.Projector(model_obj.final_feat_dim, args.proj_dim) if args.method == 'patchmix_cl' else nn.Identity()
        # 重建 criterion
        if not hasattr(args, 'label_smoothing'):
            args.label_smoothing = 0.0
        if not hasattr(args, 'use_supcon'):
            args.use_supcon = False

        if args.loss == 'ce':
            criterion = [nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()]
        elif args.method == 'patchmix_cl':
            if args.use_supcon:
                from method.supcon import PatchMixSupConLoss
                criterion = [nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(),
                             PatchMixSupConLoss(temperature=args.temperature).cuda()]
            else:
                criterion = [nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(),
                             M.PatchMixConLoss(temperature=args.temperature).cuda()]

        model_obj.cuda()
        classifier.cuda()
        projector.cuda()
        optim_params = list(model_obj.parameters()) + list(classifier.parameters()) + list(projector.parameters())
        optimizer = optim.Adam(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    # DataLoader
    train_loader = DataLoader(hard_train_dataset, batch_size=8, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    # 训练循环
    best_score = 0
    best_sp = 0
    best_se = 0
    best_epoch = 0
    epochs = kwargs.get('epochs', 5)

    for epoch in range(1, epochs + 1):
        model_obj.train()
        classifier.train()
        projector.train()

        total_loss = 0
        correct = 0
        total = 0

        for idx, (images, labels, metadata) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                features = model_obj(images)
                output = classifier(features)
                loss = criterion[0](output, labels)

                # SupCon
                if hasattr(args, 'use_supcon') and args.use_supcon:
                    proj1 = projector(features)
                    loss += args.alpha * criterion[1](proj1, proj1, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * bsz
            _, predicted = output.max(1)
            total += bsz
            correct += predicted.eq(labels).sum().item()

        # 验证
        model_obj.eval()
        classifier.eval()
        hits = [0.0] * args.n_cls
        counts = [0.0] * args.n_cls
        abn_total, abn_any = 0.0, 0.0  # 双口径

        with torch.no_grad():
            for images, labels, metadata in val_loader:
                images = images.cuda()
                labels = labels.cuda()
                with torch.cuda.amp.autocast():
                    features = model_obj(images)
                    output = classifier(features)
                _, preds = torch.max(output, 1)
                for i in range(preds.shape[0]):
                    counts[labels[i].item()] += 1.0
                    if preds[i].item() == labels[i].item():
                        hits[labels[i].item()] += 1.0
                    if labels[i].item() != 0:
                        abn_total += 1.0
                        if preds[i].item() > 0:
                            abn_any += 1.0

        sp, se, sc = get_score(hits, counts)
        se2 = abn_any / (abn_total + 1e-10) * 100
        sc2 = (sp + se2) / 2
        train_acc = 100. * correct / total
        train_loss = total_loss / total

        print(f"Epoch {epoch}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"S_p={sp:.2f}, S_e(4c)={se:.2f}, Score(4c)={sc:.2f} | S_e(2c)={se2:.2f}, Score(2c)={sc2:.2f}")

        if sc > best_score:
            best_score = sc
            best_sp = sp
            best_se = se
            best_epoch = epoch
            print(f"  ★ New best: Score={sc:.2f}")

    print(f"\n{'='*60}")
    print(f"实验结果: {experiment_name}")
    print(f"  Best Score: {best_score:.2f} (Epoch {best_epoch})")
    print(f"  Best S_p: {best_sp:.2f}, S_e: {best_se:.2f}")
    print(f"{'='*60}")

    return {
        'experiment': experiment_name,
        'best_score': best_score,
        'best_sp': best_sp,
        'best_se': best_se,
        'best_epoch': best_epoch
    }


def main():
    results = []

    # 加载基线配置
    model_args = load_model_and_data('./save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth')

    # 实验 1: 基线
    results.append(train_and_evaluate(model_args, 'baseline', epochs=5, learning_rate=5e-5))

    # 实验 2: Label Smoothing
    results.append(train_and_evaluate(model_args, 'label_smoothing',
                                      epochs=5, learning_rate=5e-5, label_smoothing=0.1))

    # 实验 3: SupCon
    results.append(train_and_evaluate(model_args, 'supcon',
                                      epochs=5, learning_rate=5e-5, use_supcon=True))

    # 实验 4: SupCon + Label Smoothing
    results.append(train_and_evaluate(model_args, 'supcon_ls',
                                      epochs=5, learning_rate=5e-5,
                                      use_supcon=True, label_smoothing=0.1))

    # 打印汇总
    print(f"\n{'='*80}")
    print(f"汇总结果")
    print(f"{'='*80}")
    print(f"{'实验':<25} {'Score':>8} {'S_p':>8} {'S_e':>8} {'vs基线':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    baseline_score = results[0]['best_score']
    for r in results:
        delta = r['best_score'] - baseline_score
        print(f"{r['experiment']:<25} {r['best_score']:>8.2f} {r['best_sp']:>8.2f} {r['best_se']:>8.2f} {delta:>+8.2f}")

    # 保存结果
    with open('sota_verify_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 sota_verify_results.json")


if __name__ == '__main__':
    main()
