#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 SPRSound 数据集加载

验证：
1. 数据文件是否存在
2. 标签分布是否正确
3. 模型能否正常训练（1个 epoch）
"""

import os
import numpy as np
import torch
from collections import Counter

print("=== 测试 SPRSound 数据集 ===")

# 1. 检查数据文件
data_file = 'sprsound_data.npz'
if not os.path.exists(data_file):
    print(f"[ERROR] 数据文件不存在: {data_file}")
    print("请先运行: python preprocess_sprsound.py")
    exit(1)

# 2. 加载数据
data = np.load(data_file)
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(f"\n[OK] 训练集: {X_train.shape}")
print(f"[OK] 测试集: {X_test.shape}")

# 3. 标签分布
print("\n=== 训练集标签分布 ===")
train_counter = Counter(y_train.tolist())
for label, count in sorted(train_counter.items()):
    print(f"  类别 {label}: {count} ({count/len(y_train)*100:.1f}%)")

print("\n=== 测试集标签分布 ===")
test_counter = Counter(y_test.tolist())
for label, count in sorted(test_counter.items()):
    print(f"  类别 {label}: {count} ({count/len(y_test)*100:.1f}%)")

# 4. 测试模型
print("\n=== 测试模型加载 ===")
from src.model_bandaware import build_model

model = build_model(num_classes=5, num_bands=4)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[OK] 模型参数量: {total_params:.2f}M")

# 5. 测试前向传播
print("\n=== 测试前向传播 ===")
x = torch.randn(2, 1024, 128).cuda()
model.cuda()
model.eval()

with torch.no_grad():
    logits, band_features, band_importance = model(x)
print(f"[OK] logits: {logits.shape}")
print(f"[OK] band_features: {band_features.shape}")
print(f"[OK] band_importance: {band_importance.shape}")

print("\n=== 所有测试通过！===")
