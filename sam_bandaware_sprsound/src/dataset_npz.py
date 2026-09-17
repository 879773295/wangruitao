"""
NPZ 格式数据集加载器（用于预处理后的 .npz 文件）
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor


class NPZDataset(Dataset):
    """
    从 .npz 文件加载数据的数据集

    Args:
        X: 音频数据 [N, samples]
        y: 标签 [N]
        processor: ASTFeatureExtractor
        augment: 是否增强（训练时用）
    """
    def __init__(self, X, y, processor, augment=False):
        self.X = X
        self.y = y
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        audio = self.X[idx].copy()
        label = self.y[idx]

        # 训练时加一点增强
        if self.augment:
            # 随机音量
            if np.random.random() < 0.3:
                vol = np.random.uniform(0.8, 1.2)
                audio = audio * vol
            # 随机偏移
            if np.random.random() < 0.3:
                shift = np.random.randint(-1600, 1600)
                audio = np.roll(audio, shift)

        # 处理成 AST 输入
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.squeeze(0)

        return input_values, label
