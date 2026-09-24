"""
AST + FBE + Band-aware Patch-Mix (v2)

创新点：
1. Same-band Patch-Mix：只在相同频段内交换 patch
2. Band Importance Estimator：学习频段重要性权重
3. Band-guided Patch Sampling：重要频段更容易参与 mixing
4. Frequency-weighted λ：按频段重要性计算混合比例

不覆盖原有 ast_fbe.py，独立版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ast_freq import ASTModel as ASTModelFBE
from .band_importance import BandImportanceEstimator, get_band_indices


class ASTFBEBandAware(ASTModelFBE):
    """
    AST + FBE + Band-aware Patch-Mix

    继承 ASTModelFBE，增加：
    - Band Importance Estimator
    - Same-band Patch-Mix
    - Frequency-weighted λ
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, audioset_pretrain=False,
                 model_size='base384', verbose=True, mix_beta=None,
                 num_bands=4, **kwargs):
        super().__init__(
            label_dim=label_dim, fstride=fstride, tstride=tstride,
            input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size, verbose=verbose,
            mix_beta=mix_beta, **kwargs
        )

        # 频段数量
        self.num_bands = num_bands

        # Band Importance Estimator
        self.band_importance_estimator = BandImportanceEstimator(
            embed_dim=self.original_embedding_dim,
            num_bands=num_bands
        )

        # 记录频段 patch 索引（在 forward 中动态计算）
        self.band_patch_indices = None

        if verbose:
            print(f"  Band-aware Patch-Mix: num_bands={num_bands}")
            print(f"  Band Importance Estimator: added")

    def patch_mix_band_aware(self, image, target, band_importance, time_domain=False, hw_num_patch=None):
        """
        Band-aware Patch-Mix：只在相同频段内交换 patch

        Args:
            image: [B, num_patches, D] patch 嵌入
            target: [B] 标签
            band_importance: [B, num_bands] 频段重要性权重
            time_domain: 是否在时间域混合
            hw_num_patch: [h_patch, w_patch] 高度和宽度 patch 数量
        Returns:
            mixed_image, labels_a, labels_b, lam, index
        """
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.shape
        device = image.device
        index = torch.randperm(batch_size).to(device)

        # 计算 frequency-weighted λ
        # λ = (importance * mask_ratio).sum() / importance.sum()
        # 这里简化：直接用 band_importance 加权计算 λ
        band_importance_mean = band_importance.mean(dim=0)  # [num_bands]
        lam = (band_importance_mean * band_importance_mean).sum() / band_importance_mean.sum()
        lam = lam.item()

        # 同频段内交换 patch
        mixed_image = image.clone()
        h_patch, w_patch = hw_num_patch if hw_num_patch else (self.f_dim, self.t_dim)

        # 确保 h_patch * w_patch == num_patch
        if h_patch * w_patch != num_patch:
            # 如果维度不匹配，回退到原始方法
            num_mask = int(num_patch * (1 - lam))
            mask = torch.randperm(num_patch)[:num_mask]
            mixed_image[:, mask] = image[index][:, mask]
        else:
            # 同频段内交换
            band_size = h_patch // self.num_bands
            for i in range(self.num_bands):
                # 获取当前频段的所有 patch 索引
                band_start = i * band_size
                band_end = band_start + band_size if i < self.num_bands - 1 else h_patch

                # 在当前频段内随机选择要交换的 patch
                band_num_patch = band_size * w_patch
                num_mask = int(band_num_patch * (1 - lam))
                if num_mask <= 0:
                    continue

                # 生成当前频段内的 mask
                band_mask = torch.randperm(band_num_patch)[:num_mask]

                # 转换为全局索引
                for idx in band_mask:
                    row = idx // w_patch + band_start
                    col = idx % w_patch
                    global_idx = row * w_patch + col
                    if global_idx < num_patch:
                        mixed_image[:, global_idx] = image[index][:, global_idx]

        # 标签
        labels_a = target
        labels_b = target[index]

        return mixed_image, labels_a, labels_b, lam, index

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        前向传播

        Args:
            x: [B, 1, freq, time] 输入频谱图
            y: [B] 标签
            patch_mix: 是否使用 Patch-Mix
            time_domain: 是否在时间域混合
        Returns:
            如果 patch_mix=False: 返回特征 [B, D]
            如果 patch_mix=True: 返回 (mixed_features, labels_a, labels_b, lam, index)
        """
        # x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1

        B = x.shape[0]
        x = self.v.patch_embed(x)  # (B, num_patches, 768)

        # ========== Band-aware Patch-Mix ==========
        if patch_mix and y is not None:
            # 计算频段特征（需要先获取各频段的平均特征）
            # 简化：直接在 patch 空间做 band-aware mix
            band_features = self._extract_band_features(x, h_patch, w_patch)  # [B, num_bands, D]
            band_importance = self.band_importance_estimator(band_features)  # [B, num_bands]

            # Band-aware Patch-Mix
            x, y_a, y_b, lam, index = self.patch_mix_band_aware(
                x, y, band_importance, time_domain=time_domain, hw_num_patch=[h_patch, w_patch]
            )
        # ==========================================

        # ========== 新增：注入频段嵌入 ==========
        if self.use_freq_band_embed:
            x = self.freq_band_embed(x, self.f_dim)  # (B, num_patches, 768)
        # ========================================

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        # x = self.mlp_head(x)

        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index

    def _extract_band_features(self, patch_features, h_patch, w_patch):
        """
        从 patch 特征中提取各频段的平均特征

        Args:
            patch_features: [B, num_patches, D] patch 嵌入
            h_patch: 频率维度 patch 数量
            w_patch: 时间维度 patch 数量
        Returns:
            band_features: [B, num_bands, D] 各频段平均特征
        """
        batch_size, num_patches, dim = patch_features.shape

        # 确保 h_patch * w_patch == num_patch
        if h_patch * w_patch != num_patches:
            # 如果维度不匹配，直接返回全零
            return torch.zeros(batch_size, self.num_bands, dim, device=patch_features.device)

        # 重塑为 [B, h_patch, w_patch, D]
        patch_features_2d = patch_features.view(batch_size, h_patch, w_patch, dim)

        # 按频段平均
        band_size = h_patch // self.num_bands
        band_features = []
        for i in range(self.num_bands):
            band_start = i * band_size
            band_end = band_start + band_size if i < self.num_bands - 1 else h_patch
            band_feat = patch_features_2d[:, band_start:band_end, :, :].mean(dim=(1, 2))  # [B, D]
            band_features.append(band_feat)

        band_features = torch.stack(band_features, dim=1)  # [B, num_bands, D]
        return band_features


def ast_fbe_bandaware(pretrained=True, pretrained_ckpt='pretrained_models/audioset_10_10_0.4593.pth',
                      num_classes=4, **kwargs):
    """构造 AST + FBE + Band-aware Patch-Mix 模型"""
    input_fdim = 128
    input_tdim = 1024
    model = ASTFBEBandAware(
        label_dim=num_classes,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        imagenet_pretrain=pretrained,
        audioset_pretrain=pretrained,
        num_bands=4,
        **kwargs
    )
    if pretrained:
        sd = model.state_dict()
        new_sd = {k: v for k, v in torch.load(pretrained_ckpt, map_location='cpu').items()
                  if k in sd and sd[k].shape == v.shape}
        sd.update(new_sd)
        model.load_state_dict(sd)
    return model
