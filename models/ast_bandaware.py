"""
原版 AST 模型 + 四个消融模块（A、B、C、D）

创新点：
A = Same-band PatchMix：同频段内交换 patch
B = Band Importance Estimator：学习频段重要性权重
C = Frequency-weighted λ：按频段重要性计算混合比例
D = BandCL Loss：约束混合后频段与源样本一致

原版模型使用 Adam 优化器，不使用 SAM
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm
from timm.models.layers import to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class BandImportanceEstimator(nn.Module):
    """频段重要性估计器（创新点 B）"""
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
    """Band-level Contrastive Consistency Loss（创新点 D）"""
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
            cos_sim = F.cosine_similarity(band_features_mix, band_features_src, dim=-1)
            return (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class ASTModelBandAware(nn.Module):
    """
    原版 AST 模型 + 四个消融模块

    创新点：
    A = Same-band PatchMix：同频段内交换 patch
    B = Band Importance Estimator：学习频段重要性权重
    C = Frequency-weighted λ：按频段重要性计算混合比例
    D = BandCL Loss：约束混合后频段与源样本一致
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384',
                 mix_beta=None, num_bands=4,
                 use_same_band_mix=True, use_band_importance=True,
                 use_freq_weighted_lam=True, use_bandcl=True,
                 band_importance_hidden_ratio=4, band_importance_temperature=1.0,
                 bandcl_loss_type='mse'):
        super(ASTModelBandAware, self).__init__()
        assert timm.__version__ == '0.4.5'

        self.mix_beta = mix_beta
        self.num_bands = num_bands

        # 创新点参数
        self.use_same_band_mix = use_same_band_mix
        self.use_band_importance = use_band_importance
        self.use_freq_weighted_lam = use_freq_weighted_lam
        self.use_bandcl = use_bandcl

        # AST 模型
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if audioset_pretrain:
            # AudioSet 预训练模型
            if not imagenet_pretrain:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')

            out_dir = './pretrained_models/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            if not os.path.exists(os.path.join(out_dir, 'audioset_10_10_0.4593.pth')):
                import wget
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out=os.path.join(out_dir, 'audioset_10_10_0.4593.pth'))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sd = torch.load(os.path.join(out_dir, 'audioset_10_10_0.4593.pth'), map_location=device)
            # 创建基础 AST 模型加载权重
            from models.ast import ASTModel
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                                   imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            self.f_dim = f_dim
            self.t_dim = t_dim

            # 位置嵌入处理（AudioSet 模型）
            # AudioSet 预训练模型：1212 patches = 12 freq × 101 time
            embed_dim = self.v.pos_embed.shape[2]
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, embed_dim).transpose(1, 2).reshape(1, embed_dim, 12, 101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            else:
                new_pos_embed = F.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, embed_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

            # 注意：AudioSet 模型的 patch_embed.proj 已经是1通道的，不需要修改

        else:
            # ImageNet 预训练或随机初始化
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # 自动获取中间形状
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            self.f_dim = f_dim
            self.t_dim = t_dim

            # 线性投影层
            new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # 位置嵌入（与原版 AST 模型保持一致）
            if imagenet_pretrain:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token)
                # reshape it to original 2D shape (14*14 for base384)
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = F.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        self.v.patch_embed.num_patches = num_patches
        self.final_feat_dim = self.original_embedding_dim

        # 分类头（与原版模型保持一致）
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim)
        )

        # 创新点 B：Band Importance Estimator
        if self.use_band_importance:
            self.band_importance_estimator = BandImportanceEstimator(
                embed_dim=self.original_embedding_dim,
                num_bands=num_bands,
                hidden_ratio=band_importance_hidden_ratio,
                temperature=band_importance_temperature
            )

        # 创新点 D：BandCL Loss
        if self.use_bandcl:
            self.bandcl_loss = BandCLLoss(loss_type=bandcl_loss_type)

    def load_sl_official_weights(self):
        """ImageNet 预训练权重已在构造函数中加载，此方法为空操作"""
        pass

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def flatten_patch(self, x):
        """将 2D patch 展平为 1D"""
        B, D, H, W = x.shape
        x = x.reshape(B, D, -1).transpose(1, 2)
        return x

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
        Same-band PatchMix（创新点 A）：只在相同频段内交换 patch

        Args:
            patch_features: [B, num_patches, D]
            lam: 混合比例（0-1）
            num_bands: 频段数量
        Returns:
            mixed_features: [B, num_patches, D]
            index: 伙伴样本索引
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

    def patch_mix_original(self, patch_features, lam):
        """
        原版 PatchMix：随机交换任意位置的 patch

        Args:
            patch_features: [B, num_patches, D]
            lam: 混合比例（0-1）
        Returns:
            mixed_features: [B, num_patches, D]
            index: 伙伴样本索引
        """
        batch_size, num_patches, dim = patch_features.shape

        # 随机选择一个伙伴样本
        index = torch.randperm(batch_size).to(patch_features.device)

        # 随机选择要交换的 patch
        num_mask = int(num_patches * (1 - lam))
        mask = torch.randperm(num_patches)[:num_mask]

        # 执行交换
        mixed_features = patch_features.clone()
        mixed_features[:, mask] = patch_features[index][:, mask]

        return mixed_features, index

    def compute_frequency_weighted_lam(self, band_importance, base_lam):
        """
        Frequency-weighted λ（创新点 C）：按频段重要性计算混合比例

        Args:
            band_importance: [B, num_bands] 频段重要性权重
            base_lam: 基础混合比例
        Returns:
            lam: [B] 加权混合比例
        """
        # 用 band_importance 加权
        band_mask_ratio = torch.ones_like(band_importance) * base_lam
        lam = (band_importance * band_mask_ratio).sum(dim=1) / band_importance.sum(dim=1)
        return lam.mean().item()

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        前向传播

        Args:
            x: [B, 1, F, T] 输入频谱
            y: [B] 标签
            patch_mix: 是否使用 PatchMix
            time_domain: 是否在时域做 PatchMix
        Returns:
            如果 patch_mix=False: 返回 features
            如果 patch_mix=True: 返回 (features, y_a, y_b, lam, index, band_features_src, band_features_mix, band_importance)
        """
        x = x.transpose(2, 3)
        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1

        B = x.shape[0]
        x = self.v.patch_embed(x)  # [B, num_patches, D]

        if patch_mix:
            # 提取原始频段特征（用于 BandCL Loss）
            band_features_src = self.extract_band_features(x, self.num_bands)  # [B, num_bands, D]

            # 使用 Beta 分布采样 lam（与原版一致）
            if self.mix_beta is not None and self.mix_beta > 0:
                lam = np.random.beta(self.mix_beta, self.mix_beta)
            else:
                lam = 0.5

            # 创新点 B：Band Importance Estimator
            band_importance = None
            if self.use_band_importance:
                band_importance = self.band_importance_estimator(band_features_src)  # [B, num_bands]

            # 创新点 C：Frequency-weighted λ
            if self.use_freq_weighted_lam and band_importance is not None:
                lam = self.compute_frequency_weighted_lam(band_importance, lam)

            # 创新点 A：Same-band PatchMix 或原版 PatchMix
            if self.use_same_band_mix:
                x, index = self.patch_mix_band_aware(x, lam, self.num_bands)
            else:
                x, index = self.patch_mix_original(x, lam)

            # 提取混合后的频段特征（用于 BandCL Loss）
            band_features_mix = self.extract_band_features(x, self.num_bands)  # [B, num_bands, D]

            # 标签混合
            y_a, y_b = y, y[index]
        else:
            band_features_src = None
            band_features_mix = None
            band_importance = None
            index = None

        # Transformer 编码
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index, band_features_src, band_features_mix, band_importance

    def compute_bandcl_loss(self, band_features_src, band_features_mix):
        """计算 BandCL Loss（创新点 D）"""
        if self.use_bandcl and band_features_src is not None and band_features_mix is not None:
            return self.bandcl_loss(band_features_src, band_features_mix)
        else:
            return torch.tensor(0.0).to(band_features_src.device if band_features_src is not None else 'cuda')
