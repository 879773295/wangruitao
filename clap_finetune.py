#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLAP 微调：在 ICBHI 上微调 CLAP 模型

CLAP 优势：
1. 音频-文本对比学习（更通用的特征）
2. 支持零样本分类
3. 预训练在大规模音频-文本对上

微调策略：
1. 冻结 CLAP 音频编码器
2. 只训练分类头
3. 或者端到端微调（更慢但效果更好）
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ClapModel, ClapProcessor

import main as M
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score


class CLAPClassifier(nn.Module):
    """CLAP + 分类头"""
    def __init__(self, clap_model, num_classes=4, freeze_clap=True):
        super().__init__()
        self.clap = clap_model
        self.freeze_clap = freeze_clap

        # 冻结 CLAP
        if freeze_clap:
            for param in self.clap.parameters():
                param.requires_grad = False

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio, sampling_rate=48000):
        """
        audio: [B, num_samples] 原始音频
        """
        # CLAP 提取特征
        with torch.no_grad() if self.freeze_clap else torch.enable_grad():
            audio_features = self.clap.get_audio_features(audio=audio, sampling_rate=sampling_rate)

        # 分类
        logits = self.classifier(audio_features)
        return logits


def main():
    print("=== CLAP 微调 ===")

    # 1. 加载 CLAP
    print("\n1. 加载 CLAP 模型...")
    clap_path = './pretrained_models/models/laion--clap-htsat-fused/snapshots/master'
    clap_model = ClapModel.from_pretrained(clap_path)
    processor = ClapProcessor.from_pretrained(clap_path)
    print(f"CLAP 参数: {sum(p.numel() for p in clap_model.parameters())/1e6:.1f}M")

    # 2. 构建分类器
    print("\n2. 构建分类器...")
    model = CLAPClassifier(clap_model, num_classes=4, freeze_clap=True)
    model.cuda()
    print(f"总参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    # 3. 加载数据
    print("\n3. 加载 ICBHI 数据...")
    # 需要加载原始音频，不是 mel-spectrogram
    # 这里简化处理，实际需要修改 dataset

    # 4. 训练
    print("\n4. 训练...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n✅ CLAP 微调框架已搭建！")
    print("下一步：修改 dataset 加载原始音频，然后训练")


if __name__ == '__main__':
    main()
