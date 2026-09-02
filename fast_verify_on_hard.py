#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在难样本上快速验证 SOTA 模块

用难样本小数据集训练，对比不同 SOTA 模块的效果。
难样本更敏感，能快速判断模块是否有效。
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from copy import deepcopy

import main as M
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment


def load_model_and_data(ckpt_path):
    """加载模型和训练集"""
    print(f"=== 加载模型 {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    saved_args = ckpt['args']

    # 补丁：旧 checkpoint 缺少新参数
    if not hasattr(saved_args, 'label_smoothing'):
        saved_args.label_smoothing = 0.0
    if not hasattr(saved_args, 'use_supcon'):
        saved_args.use_supcon = False

    # 构建模型
    model, classifier, projector, criterion, optimizer = M.set_model(saved_args)
    model.load_state_dict(ckpt['model'])
    classifier.load_state_dict(ckpt['classifier'])

    return model, classifier, projector, saved_args


def build_datasets(args, hard_indices_path='hard_train_indices.json'):
    """构建难样本训练集和验证集"""
    print(f"\n=== 构建数据集 ===")

    # 加载难样本索引
    with open(hard_indices_path, 'r') as f:
        hard_samples = json.load(f)
    hard_indices = [s['index'] for s in hard_samples]
    print(f"难样本数量: {len(hard_indices)}")

    # 构建完整训练集
    train_transform = [transforms.ToTensor(),
                       transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    train_transform = transforms.Compose(train_transform)
    full_train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=False)

    # 构建难样本子集
    hard_train_dataset = Subset(full_train_dataset, hard_indices)

    # 构建验证集（全量）
    val_transform = [transforms.ToTensor(),
                     transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    val_transform = transforms.Compose(val_transform)
    val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=False)

    print(f"难样本训练集: {len(hard_train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    return hard_train_dataset, val_dataset


def train_one_epoch(model, classifier, projector, criterion, optimizer, scaler, dataloader, epoch, args):
    """训练一个 epoch"""
    model.train()
    classifier.train()
    projector.train()

    total_loss = 0
    correct = 0
    total = 0

    for idx, (images, labels, metadata) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                features = model(images)
                output = classifier(features)
                loss = criterion[0](output, labels)
            elif args.method == 'patchmix_cl':
                features = model(images)
                output = classifier(features)
                loss = criterion[0](output, labels)

                # SupCon 或 PatchMixConLoss
                if args.use_supcon:
                    proj1 = projector(features)
                    loss += args.alpha * criterion[1](proj1, proj1, labels)
                else:
                    # 简化：只用分类损失（快速验证）
                    pass

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * bsz
        _, predicted = output.max(1)
        total += bsz
        correct += predicted.eq(labels).sum().item()

        if (idx + 1) % 100 == 0:
            print(f"  Epoch {epoch}, batch {idx + 1}/{len(dataloader)}, "
                  f"loss: {loss.item():.4f}, acc: {100.*correct/total:.2f}%")

    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc


def evaluate(model, classifier, val_loader, args):
    """在验证集上评估 Score"""
    model.eval()
    classifier.eval()

    hits = [0.0] * args.n_cls
    counts = [0.0] * args.n_cls

    with torch.no_grad():
        for images, labels, metadata in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            with torch.cuda.amp.autocast():
                features = model(images)
                output = classifier(features)

            _, preds = torch.max(output, 1)
            for i in range(preds.shape[0]):
                counts[labels[i].item()] += 1.0
                if preds[i].item() == labels[i].item():
                    hits[labels[i].item()] += 1.0

    sp, se, sc = get_score(hits, counts)
    return sp, se, sc


def run_experiment(args, experiment_name, use_label_smoothing=False, use_supcon=False, lr=5e-5, epochs=10):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"  Label Smoothing: {use_label_smoothing}")
    print(f"  SupCon: {use_supcon}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"{'='*60}")

    # 加载模型和数据
    model, classifier, projector, model_args = load_model_and_data(
        './save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth'
    )
    hard_train_dataset, val_dataset = build_datasets(model_args)

    # 设置参数
    args.label_smoothing = 0.1 if use_label_smoothing else 0.0
    args.use_supcon = use_supcon
    args.epochs = epochs
    args.learning_rate = lr
    # 补丁：确保必需参数存在
    if not hasattr(args, 'loss'):
        args.loss = 'ce'
    if not hasattr(args, 'focal_gamma'):
        args.focal_gamma = 2.0
    if not hasattr(args, 'weighted_loss'):
        args.weighted_loss = False
    if not hasattr(args, 'alpha'):
        args.alpha = 1.0
    if not hasattr(args, 'temperature'):
        args.temperature = 0.06
    if not hasattr(args, 'method'):
        args.method = 'patchmix_cl'
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 1e-6
    if not hasattr(args, 'proj_dim'):
        args.proj_dim = 768
    # 从 model_args 复制
    args.n_cls = model_args.n_cls
    args.cls_list = model_args.cls_list
    args.h = model_args.h
    args.w = model_args.w
    args.resz = model_args.resz

    # 重建 criterion（根据新参数）
    if not hasattr(args, 'weighted_loss') or not args.weighted_loss:
        weights = None
    else:
        weights = torch.tensor(model_args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    elif args.loss == 'focal':
        criterion = M.FocalLoss(gamma=args.focal_gamma, alpha=weights)

    if args.method == 'ce':
        criterion = [criterion.cuda()]
    elif args.method == 'patchmix_cl':
        if args.use_supcon:
            from method.supcon import PatchMixSupConLoss
            criterion = [criterion.cuda(), PatchMixSupConLoss(temperature=args.temperature).cuda()]
        else:
            criterion = [criterion.cuda(), M.PatchMixConLoss(temperature=args.temperature).cuda()]

    # 创建 DataLoader
    train_loader = DataLoader(hard_train_dataset, batch_size=8, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 优化器
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    optimizer = optim.Adam(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    best_score = 0
    best_sp = 0
    best_se = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, classifier, projector, criterion, optimizer, scaler, train_loader, epoch, args
        )

        # 验证
        sp, se, sc = evaluate(model, classifier, val_loader, args)

        print(f"Epoch {epoch}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"S_p={sp:.2f}, S_e={se:.2f}, Score={sc:.2f}")

        # 保存最佳
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
    import argparse
    parser = argparse.ArgumentParser('难样本 SOTA 模块验证')
    parser.add_argument('--epochs', type=int, default=10, help='每个实验训练 epoch 数')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    args = parser.parse_args()

    results = []

    # 基线：原始 CE loss
    results.append(run_experiment(args, 'baseline_ce', epochs=args.epochs, lr=args.lr))

    # 实验 1：Label Smoothing
    results.append(run_experiment(args, 'label_smoothing_0.1',
                                  use_label_smoothing=True, epochs=args.epochs, lr=args.lr))

    # 实验 2：SupCon
    results.append(run_experiment(args, 'supcon',
                                  use_supcon=True, epochs=args.epochs, lr=args.lr))

    # 实验 3：Label Smoothing + SupCon
    results.append(run_experiment(args, 'ls01_supcon',
                                  use_label_smoothing=True, use_supcon=True,
                                  epochs=args.epochs, lr=args.lr))

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"汇总结果")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['experiment']:30s} | Score={r['best_score']:.2f} | S_p={r['best_sp']:.2f} | S_e={r['best_se']:.2f}")

    # 保存结果
    with open('hard_sample_sota_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 hard_sample_sota_results.json")


if __name__ == '__main__':
    main()
