"""
SAM + Band-aware Patch-Mix 模型（7 分类版，支持四模块消融）

移植自 sam_bandaware_sprsound/src/model_bandaware.py（SPRSound 5 类版），
底座同为 HuggingFace AST，仅 num_classes 默认值改为 7。

创新点（全部实现，可独立开关）：
1. A: Same-band Patch-Mix —— 只在相同频段内交换 patch
2. B: Band Importance Estimator —— 学习频段重要性权重
3. C: Frequency-weighted λ —— 按频段重要性计算混合比例
4. D: BandCL Loss —— 约束混合后频段与源样本一致

参数控制消融（在 train_event.py 中传入）：
--use_same_band_mix     启用 A
--use_band_importance   启用 B
--use_freq_weighted_lam 启用 C
--use_bandcl            启用 D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import ASTModel


class BandImportanceEstimator(nn.Module):
    """频段重要性估计器"""
    def __init__(self, embed_dim=768, num_bands=4, hidden_ratio=4, temperature=1.0):
        super().__init__()
        self.num_bands = num_bands
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // hidden_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim // hidden_ratio, 1),
        )

    def forward(self, band_features):
        """
        Args:
            band_features: [B, num_bands, D]
        Returns:
            band_importance: [B, num_bands]
        """
        importance_scores = self.mlp(band_features)  # [B, num_bands, 1]
        importance_scores = importance_scores.squeeze(-1) / self.temperature  # [B, num_bands]
        band_importance = F.softmax(importance_scores, dim=-1)  # [B, num_bands]
        return band_importance


class BandCLLoss(nn.Module):
    """Band-level Contrastive Consistency Loss"""
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, band_features_src, band_features_mix):
        """
        Args:
            band_features_src: [B, num_bands, D]
            band_features_mix: [B, num_bands, D]
        Returns:
            loss: scalar
        """
        if self.loss_type == 'mse':
            return F.mse_loss(band_features_mix, band_features_src, reduction='mean')
        elif self.loss_type == 'l1':
            return F.l1_loss(band_features_mix, band_features_src, reduction='mean')
        elif self.loss_type == 'cosine':
            # 余弦相似度损失（1 - cosine_similarity）
            cos_sim = F.cosine_similarity(band_features_mix, band_features_src, dim=-1)
            return (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class CustomASTBandAware(nn.Module):
    """
    SAM + Band-aware Patch-Mix 模型（完整实现，支持消融）

    创新点（全部实现，可独立开关）：
    1. Same-band Patch-Mix：只在相同频段内交换 patch
    2. Band Importance Estimator：学习频段重要性权重
    3. Frequency-weighted λ：按频段重要性计算混合比例
    4. BandCL Loss：约束混合后频段与源样本一致
    """
    def __init__(self, num_classes=7, num_bands=4,
                 use_same_band_mix=True, use_band_importance=True,
                 use_freq_weighted_lam=True, use_bandcl=True,
                 band_importance_hidden_ratio=4, band_importance_temperature=1.0,
                 bandcl_loss_type='mse'):
        super().__init__()
        # AST 模型（HuggingFace）
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

        # 创新点参数
        self.num_bands = num_bands
        self.use_same_band_mix = use_same_band_mix
        self.use_band_importance = use_band_importance
        self.use_freq_weighted_lam = use_freq_weighted_lam
        self.use_bandcl = use_bandcl

        # 创新点 2：Band Importance Estimator（可选）
        if self.use_band_importance:
            self.band_importance_estimator = BandImportanceEstimator(
                embed_dim=768,
                num_bands=num_bands,
                hidden_ratio=band_importance_hidden_ratio,
                temperature=band_importance_temperature
            )

        # 创新点 4：BandCL Loss（可选）
        if self.use_bandcl:
            self.bandcl_loss = BandCLLoss(loss_type=bandcl_loss_type)

    def extract_band_features(self, patch_features, num_bands):
        """
        从 patch 特征中提取各频段的平均特征

        Args:
            patch_features: [B, num_patches, D]
            num_bands: 频段数量
        Returns:
            band_features: [B, num_bands, D]
        """
        batch_size, num_patches, dim = patch_features.shape

        # 计算每个频段的 patch 数量
        band_size = num_patches // num_bands
        band_features = []

        for i in range(num_bands):
            start_idx = i * band_size
            end_idx = start_idx + band_size if i < num_bands - 1 else num_patches
            band_feat = patch_features[:, start_idx:end_idx, :].mean(dim=1)  # [B, D]
            band_features.append(band_feat)

        band_features = torch.stack(band_features, dim=1)  # [B, num_bands, D]
        return band_features

    def patch_mix_band_aware(self, patch_features, lam, num_bands):
        """
        Same-band Patch-Mix：只在相同频段内交换 patch

        Args:
            patch_features: [B, num_patches, D]
            lam: 混合比例（0-1）
            num_bands: 频段数量
        Returns:
            mixed_features: [B, num_patches, D]
        """
        batch_size, num_patches, dim = patch_features.shape
        band_size = num_patches // num_bands

        # 随机选择一个伙伴样本
        index = torch.randperm(batch_size).to(patch_features.device)

        mixed_features = patch_features.clone()

        # 同频段内交换
        for i in range(num_bands):
            start_idx = i * band_size
            end_idx = start_idx + band_size if i < num_bands - 1 else num_patches
            band_num_patch = end_idx - start_idx

            # 在当前频段内随机选择要交换的 patch 数量
            num_mask = int(band_num_patch * (1 - lam))
            if num_mask <= 0:
                continue

            # 生成当前频段内的 mask
            band_mask = torch.randperm(band_num_patch)[:num_mask]

            # 执行交换
            for idx in band_mask:
                global_idx = start_idx + idx
                if global_idx < num_patches:
                    mixed_features[:, global_idx] = patch_features[index][:, global_idx]

        return mixed_features, index

    def compute_frequency_weighted_lam(self, band_importance, band_mask_ratio):
        """
        Frequency-weighted λ：按频段重要性计算混合比例

        Args:
            band_importance: [B, num_bands] 频段重要性权重
            band_mask_ratio: [num_bands] 每个频段的 mask 比例
        Returns:
            lam: [B] 加权混合比例
        """
        # λ = (importance * mask_ratio).sum() / importance.sum()
        lam = (band_importance * band_mask_ratio).sum(dim=1) / band_importance.sum(dim=1)
        return lam

    def forward(self, x, patch_mix=False, mix_lam=0.5):
        """
        前向传播

        Args:
            x: [B, num_samples] 原始音频
            patch_mix: 是否使用 Patch-Mix
            mix_lam: Patch-Mix 的基础混合比例
        Returns:
            如果 patch_mix=False: 返回 (logits, band_features, band_importance)
            如果 patch_mix=True: 返回 (logits, band_features_src, band_features_mix, band_importance, index)
        """
        # AST 前向传播
        outputs = self.ast(x)
        embeddings = outputs.last_hidden_state  # [B, num_patches, D]

        # 提取频段特征（用于 Band Importance Estimator 和 BandCL Loss）
        band_features_src = self.extract_band_features(embeddings, self.num_bands)  # [B, num_bands, D]

        # 创新点 2：Band Importance Estimator（可选）
        band_importance = None
        if self.use_band_importance:
            band_importance = self.band_importance_estimator(band_features_src)  # [B, num_bands]

        # 创新点 3：Frequency-weighted λ（可选）
        if patch_mix:
            if self.use_freq_weighted_lam and band_importance is not None:
                # 用 band_importance 加权计算 λ
                band_mask_ratio = torch.ones(self.num_bands).to(x.device) * mix_lam
                lam = self.compute_frequency_weighted_lam(band_importance, band_mask_ratio)  # [B]
                lam = lam.mean().item()  # 取平均
            else:
                lam = mix_lam

            # 创新点 1：Same-band Patch-Mix（可选）
            if self.use_same_band_mix:
                mixed_embeddings, index = self.patch_mix_band_aware(embeddings, lam, self.num_bands)
            else:
                # 原始 PatchMix（随机交换）
                batch_size = embeddings.shape[0]
                num_patches = embeddings.shape[1]
                num_mask = int(num_patches * (1 - lam))
                index = torch.randperm(batch_size).to(embeddings.device)
                mixed_embeddings = embeddings.clone()
                mask = torch.randperm(num_patches)[:num_mask]
                mixed_embeddings[:, mask] = embeddings[index][:, mask]

            # 提取混合后的频段特征
            band_features_mix = self.extract_band_features(mixed_embeddings, self.num_bands)  # [B, num_bands, D]

            # 分类
            pooled = mixed_embeddings.mean(dim=1)  # [B, D]
            logits = self.classifier(pooled)  # [B, num_classes]

            return logits, band_features_src, band_features_mix, band_importance, index
        else:
            # 分类
            pooled = embeddings.mean(dim=1)  # [B, D]
            logits = self.classifier(pooled)  # [B, num_classes]

            return logits, band_features_src, band_importance

    def compute_bandcl_loss(self, band_features_src, band_features_mix):
        """计算 BandCL Loss"""
        if self.use_bandcl:
            return self.bandcl_loss(band_features_src, band_features_mix)
        else:
            return torch.tensor(0.0).to(band_features_src.device)


def build_model(num_classes=7, num_bands=4,
                use_same_band_mix=True, use_band_importance=True,
                use_freq_weighted_lam=True, use_bandcl=True,
                band_importance_hidden_ratio=4, band_importance_temperature=1.0,
                bandcl_loss_type='mse'):
    """构建模型（支持消融）"""
    model = CustomASTBandAware(
        num_classes=num_classes,
        num_bands=num_bands,
        use_same_band_mix=use_same_band_mix,
        use_band_importance=use_band_importance,
        use_freq_weighted_lam=use_freq_weighted_lam,
        use_bandcl=use_bandcl,
        band_importance_hidden_ratio=band_importance_hidden_ratio,
        band_importance_temperature=band_importance_temperature,
        bandcl_loss_type=bandcl_loss_type
    )
    return model
