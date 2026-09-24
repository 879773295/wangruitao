"""
Band Importance Estimator
估计每个频段的重要性权重，用于 Band-guided Patch Sampling 和 Frequency-weighted λ

创新点：
- 用 MLP + softmax 学习频段重要性
- 重要频段的 patch 更容易参与 mixing
- 混合比例 λ 按频段重要性加权
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandImportanceEstimator(nn.Module):
    """
    频段重要性估计器

    输入: 各频段特征 [B, num_bands, D]
    输出: 频段重要性权重 [B, num_bands]（softmax 归一化）
    """
    def __init__(self, embed_dim=768, num_bands=4):
        super().__init__()
        self.num_bands = num_bands

        # MLP: 嵌入维度 → 1（每个频段一个重要性分数）
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, band_features):
        """
        Args:
            band_features: [B, num_bands, D] 各频段特征
        Returns:
            band_importance: [B, num_bands] 频段重要性权重（softmax 归一化）
        """
        # band_features: [B, num_bands, D]
        importance_scores = self.mlp(band_features)  # [B, num_bands, 1]
        importance_scores = importance_scores.squeeze(-1)  # [B, num_bands]

        # softmax 归一化
        band_importance = F.softmax(importance_scores, dim=-1)  # [B, num_bands]

        return band_importance


def get_band_indices(num_freq_patches, f_dim, num_bands=4):
    """
    根据频段边界划分 patch 索引

    Args:
        num_freq_patches: 频率维度的 patch 数量
        f_dim: 每个频段的 patch 数量（num_freq_patches / num_bands）
        num_bands: 频段数量
    Returns:
        band_indices: list of tensor，每个元素是该频段的 patch 索引
    """
    band_indices = []
    band_size = num_freq_patches // num_bands

    for i in range(num_bands):
        start_idx = i * band_size
        end_idx = start_idx + band_size if i < num_bands - 1 else num_freq_patches
        band_indices.append(torch.arange(start_idx, end_idx))

    return band_indices
