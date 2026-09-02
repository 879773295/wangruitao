#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建难样本小数据集（从训练集提取）

用当前 best model 对训练集推理，找出难样本（预测错误 + 置信度低），
构建小数据集用于快速验证 SOTA 模块。
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import main as M
from util.icbhi_dataset import ICBHIDataset
from util.augmentation import SpecAugment


def load_model(ckpt_path):
    """加载模型"""
    print(f"=== 加载模型 {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    saved_args = ckpt['args']

    # 补丁：旧 checkpoint 缺少新参数
    if not hasattr(saved_args, 'label_smoothing'):
        saved_args.label_smoothing = 0.0
    if not hasattr(saved_args, 'use_supcon'):
        saved_args.use_supcon = False

    model, classifier, projector, criterion, optimizer = M.set_model(saved_args)
    model.load_state_dict(ckpt['model'])
    classifier.load_state_dict(ckpt['classifier'])
    model.eval()
    classifier.eval()

    return model, classifier, saved_args


def build_train_dataset(args):
    """构建训练集"""
    print(f"\n=== 构建训练集 ===")
    train_transform = [transforms.ToTensor(),
                       transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    train_transform = transforms.Compose(train_transform)
    train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
    return train_dataset


def extract_hard_train_samples(model, classifier, train_dataset, args, top_k=200):
    """从训练集中提取难样本"""
    print(f"\n=== 提取训练集难样本 (top_k={top_k}) ===")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    predictions = []
    true_labels = []
    confidences = []
    all_indices = []

    with torch.no_grad():
        for idx, (images, labels, metadata) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            with torch.cuda.amp.autocast():
                features = model(images)
                output = classifier(features)

            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()

            predictions.append(pred)
            true_labels.append(labels.item())
            confidences.append(conf)
            all_indices.append(idx)

            if (idx + 1) % 500 == 0:
                print(f"  处理 {idx + 1}/{len(train_dataset)} 样本...")

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)

    # 统计
    acc = (predictions == true_labels).mean() * 100
    print(f"\n训练集准确率: {acc:.2f}%")
    print(f"类别分布: normal={np.sum(true_labels==0)}, crackle={np.sum(true_labels==1)}, "
          f"wheeze={np.sum(true_labels==2)}, both={np.sum(true_labels==3)}")

    # 找预测错误的样本
    wrong_mask = predictions != true_labels
    print(f"预测错误: {np.sum(wrong_mask)} 样本")

    # 找置信度最低的样本
    low_conf_indices = np.argsort(confidences)[:top_k]

    # 合并
    hard_indices = np.unique(np.concatenate([
        np.where(wrong_mask)[0],
        low_conf_indices
    ]))

    print(f"难样本总数: {len(hard_indices)}")
    print(f"  - 预测错误: {np.sum(wrong_mask)}")
    print(f"  - 置信度低 (top {top_k}): {len(low_conf_indices)}")
    print(f"  - 去重后: {len(hard_indices)}")

    # 统计难样本的类别分布
    hard_labels = true_labels[hard_indices]
    print(f"\n难样本类别分布:")
    for cls in range(args.n_cls):
        count = np.sum(hard_labels == cls)
        print(f"  Class {cls} ({args.cls_list[cls]}): {count}")

    return hard_indices, predictions, true_labels, confidences


def save_hard_dataset(train_dataset, hard_indices, save_path='hard_train_indices.json'):
    """保存难样本索引"""
    print(f"\n=== 保存难样本索引到 {save_path} ===")

    hard_samples = []
    for idx in hard_indices:
        sample = train_dataset.audio_data[idx]
        label = sample[1]

        hard_samples.append({
            'index': int(idx),
            'label': int(label),
        })

    with open(save_path, 'w') as f:
        json.dump(hard_samples, f, indent=2)

    print(f"保存 {len(hard_samples)} 个难样本索引")
    return hard_samples


def main():
    import argparse
    parser = argparse.ArgumentParser('构建难样本小数据集')
    parser.add_argument('--ckpt', type=str,
                        default='./save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth',
                        help='模型 checkpoint 路径')
    parser.add_argument('--top_k', type=int, default=200,
                        help='提取置信度最低的 top_k 个样本')
    parser.add_argument('--save_path', type=str, default='hard_train_indices.json',
                        help='保存路径')
    args = parser.parse_args()

    # 加载模型
    model, classifier, model_args = load_model(args.ckpt)

    # 构建训练集
    train_dataset = build_train_dataset(model_args)

    # 提取难样本
    hard_indices, predictions, true_labels, confidences = extract_hard_train_samples(
        model, classifier, train_dataset, model_args, top_k=args.top_k
    )

    # 保存
    save_hard_dataset(train_dataset, hard_indices, args.save_path)

    # 创建 Subset 用于快速验证
    hard_subset = Subset(train_dataset, hard_indices.tolist())
    print(f"\n=== 难样本小数据集创建成功 ===")
    print(f"原始训练集: {len(train_dataset)} 样本")
    print(f"难样本子集: {len(hard_subset)} 样本")
    print(f"比例: {len(hard_subset)/len(train_dataset)*100:.1f}%")

    print(f"\n下一步：用 hard_train_indices.json 加载难样本子集，快速验证 SOTA 模块")


if __name__ == '__main__':
    main()
