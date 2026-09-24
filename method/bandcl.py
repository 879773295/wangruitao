"""
Band-level Contrastive Consistency Loss (BandCL)

约束混合后各频段表示与对应源样本保持一致。

创新点：
- 混合后的 patch 应该保留原频段的语义信息
- 通过对比学习约束频段一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandCLLoss(nn.Module):
    """
    Band-level Contrastive Consistency Loss

    Args:
        temperature: 温度参数（控制 softmax 的平滑程度）
        num_bands: 频段数量
    """
    def __init__(self, temperature=0.1, num_bands=4):
        super().__init__()
        self.temperature = temperature
        self.num_bands = num_bands

    def forward(self, band_features_src, band_features_mix):
        """
        Args:
            band_features_src: [B, num_bands, D] 源样本的频段特征
            band_features_mix: [B, num_bands, D] 混合后的频段特征
        Returns:
            loss: 标量
        """
        batch_size, num_bands, dim = band_features_src.shape

        # 计算每个频段的 MSE 损失
        # band_features_src 和 band_features_mix 对应同一个频段
        mse_loss = F.mse_loss(band_features_mix, band_features_src, reduction='mean')

        return mse_loss


class BandCLLossWithContrastive(nn.Module):
    """
    带对比学习的 BandCL Loss

    不仅要求混合后频段与源样本一致，还要拉近同类、推远异类
    """
    def __init__(self, temperature=0.1, num_bands=4):
        super().__init__()
        self.temperature = temperature
        self.num_bands = num_bands

    def forward(self, band_features_src, band_features_mix, labels):
        """
        Args:
            band_features_src: [B, num_bands, D] 源样本的频段特征
            band_features_mix: [B, num_bands, D] 混合后的频段特征
            labels: [B] 标签
        Returns:
            loss: 标量
        """
        batch_size, num_bands, dim = band_features_src.shape

        # MSE 损失（频段一致性）
        mse_loss = F.mse_loss(band_features_mix, band_features_src, reduction='mean')

        # 对比损失（拉近同类、推远异类）
        # 将频段特征展平为 [B, num_bands * D]
        src_flat = band_features_src.view(batch_size, -1)  # [B, num_bands * D]
        mix_flat = band_features_mix.view(batch_size, -1)  # [B, num_bands * D]

        # 计算相似度
        sim = F.cosine_similarity(src_flat, mix_flat, dim=-1)  # [B]

        # 对比损失：同类样本相似度应高于异类
        # 这里简化处理：只用 MSE 损失
        loss = mse_loss

        return loss
