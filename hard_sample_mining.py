#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
难样本挖掘脚本

用当前 best model 对验证集推理，找出难样本（预测错误 + 置信度低），
构建小数据集用于快速验证 SOTA 模块。
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import main as M
from util.icbhi_dataset import ICBHIDataset
from util.augmentation import SpecAugment


def load_model_and_data(ckpt_path, args):
    """加载模型和验证集"""
    print(f"=== 加载模型 {ckpt_path} ===")

    # 设置 args（从 checkpoint 里读取）
    ckpt = torch.load(ckpt_path, map_location='cpu')
    saved_args = ckpt['args']

    # 补丁：旧 checkpoint 缺少新参数，手动设置默认值
    if not hasattr(saved_args, 'label_smoothing'):
        saved_args.label_smoothing = 0.0
    if not hasattr(saved_args, 'use_supcon'):
        saved_args.use_supcon = False

    # 用 saved_args 构建 model
    model, classifier, projector, criterion, optimizer = M.set_model(saved_args)
    model.load_state_dict(ckpt['model'])
    classifier.load_state_dict(ckpt['classifier'])
    model.eval()
    classifier.eval()

    # 构建验证集
    saved_args.batch_size = 1
    saved_args.num_workers = 0
    val_transform = [transforms.ToTensor(),
                     transforms.Resize(size=(int(saved_args.h * saved_args.resz),
                                             int(saved_args.w * saved_args.resz)))]
    val_transform = transforms.Compose(val_transform)
    val_dataset = ICBHIDataset(train_flag=False, transform=val_transform,
                               args=saved_args, print_flag=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    return model, classifier, val_dataset, val_loader, saved_args


def extract_hard_samples(model, classifier, val_loader, val_dataset, args, top_k=500):
    """提取难样本：预测错误 + 置信度最低"""
    print(f"\n=== 提取难样本 (top_k={top_k}) ===")

    predictions = []
    true_labels = []
    confidences = []

    with torch.no_grad():
        for idx, (images, labels, metadata) in enumerate(val_loader):
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

            if (idx + 1) % 500 == 0:
                print(f"  处理 {idx + 1}/{len(val_loader)} 样本...")

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)

    # 统计
    acc = (predictions == true_labels).mean() * 100
    print(f"\n验证集准确率: {acc:.2f}%")
    print(f"类别分布: normal={np.sum(true_labels==0)}, crackle={np.sum(true_labels==1)}, "
          f"wheeze={np.sum(true_labels==2)}, both={np.sum(true_labels==3)}")

    # 找预测错误的样本
    wrong_mask = predictions != true_labels
    print(f"预测错误: {np.sum(wrong_mask)} 样本")

    # 找置信度最低的样本（不管对错）
    low_conf_indices = np.argsort(confidences)[:top_k]

    # 合并：预测错误 + 置信度低（去重）
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


def save_hard_samples(val_dataset, hard_indices, save_path='hard_samples.json'):
    """保存难样本信息"""
    print(f"\n=== 保存难样本索引到 {save_path} ===")

    hard_samples = []
    for idx in hard_indices:
        # audio_data 的结构是 (audio_chunk, label, metadata)
        sample = val_dataset.audio_data[idx]
        label = sample[1]
        metadata = val_dataset.metadata[idx]
        filename = val_dataset.filenames[idx] if idx < len(val_dataset.filenames) else f"sample_{idx}"

        hard_samples.append({
            'index': int(idx),
            'label': int(label),
            'metadata': metadata.tolist() if torch.is_tensor(metadata) else metadata,
        })

    with open(save_path, 'w') as f:
        json.dump(hard_samples, f, indent=2)

    print(f"保存 {len(hard_samples)} 个难样本索引")
    return hard_samples


def main():
    import argparse
    parser = argparse.ArgumentParser('难样本挖掘')
    parser.add_argument('--ckpt', type=str,
                        default='./save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth',
                        help='模型 checkpoint 路径')
    parser.add_argument('--top_k', type=int, default=500,
                        help='提取置信度最低的 top_k 个样本')
    parser.add_argument('--save_path', type=str, default='hard_samples.json',
                        help='保存路径')
    args = parser.parse_args()

    # 加载模型和数据
    model, classifier, val_dataset, val_loader, model_args = load_model_and_data(args.ckpt, None)

    # 提取难样本
    hard_indices, predictions, true_labels, confidences = extract_hard_samples(
        model, classifier, val_loader, val_dataset, model_args, top_k=args.top_k
    )

    # 保存
    save_hard_samples(val_dataset, hard_indices, args.save_path)

    # 打印一些难样本的例子
    print(f"\n=== 难样本示例 ===")
    for i in range(min(10, len(hard_indices))):
        idx = hard_indices[i]
        label = true_labels[idx]
        pred = predictions[idx]
        conf = confidences[idx]
        print(f"  Index {idx}: True={model_args.cls_list[label]}, Pred={model_args.cls_list[pred]}, Conf={conf:.4f}")

    print(f"\n完成！难样本索引已保存到 {args.save_path}")


if __name__ == '__main__':
    main()
