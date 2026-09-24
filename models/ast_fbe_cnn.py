"""
FBE + CNN 二通道模型
==========================
创新点：
1. FBE：全局特征 + 频段特征（已包含 AST）
2. CNN：局部特征

融合：Concat(fbe_feat, cnn_feat) → 分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ast_freq import ASTModel as ASTModelFBE
from .ast_cnn_dual import CNNBranch


class ASTFBECNN(nn.Module):
    """
    FBE + CNN 二通道模型

    FBE 已经包含 AST + 频段嵌入，所以：
    通道1（FBE）：全局特征 + 频段特征 [B, 768]
    通道2（CNN）：局部特征 [B, 256]

    融合：[B, 768+256] = [B, 1024] → 分类
    """
    def __init__(self, label_dim=4, input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, audioset_pretrain=True,
                 cnn_embed_dim=256, verbose=True):
        super().__init__()

        # 通道1：FBE（全局特征 + 频段特征）
        if verbose:
            print("=== 通道1：FBE（全局+频段特征）===")
        self.fbe = ASTModelFBE(
            label_dim=label_dim,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            use_freq_band_embed=True,
            verbose=verbose
        )
        fbe_dim = self.fbe.final_feat_dim  # 768

        # 通道2：CNN（局部特征）
        if verbose:
            print("\n=== 通道2：CNN（局部特征）===")
        self.cnn = CNNBranch(cnn_embed_dim=cnn_embed_dim)
        cnn_dim = cnn_embed_dim  # 256

        # 融合分类头
        total_dim = fbe_dim + cnn_dim  # 768 + 256 = 1024
        if verbose:
            print(f"\n=== 融合：{total_dim} → {label_dim} ===")
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, label_dim)
        )

        # 更新特征维度
        self.final_feat_dim = total_dim

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        二通道前向传播
        :param x: (B, 1, freq, time)
        :return: 分类 logits 或 (logits, y_a, y_b, lam, index)
        """
        # 通道1：FBE（包含 AST + 频段嵌入）
        fbe_feat = self.fbe(x)  # [B, 768]

        # 通道2：CNN
        cnn_feat = self.cnn(x)  # [B, 256]

        # 融合
        fused = torch.cat([fbe_feat, cnn_feat], dim=1)  # [B, 1024]

        if not patch_mix:
            return fused
        else:
            return fused, y, y, 1.0, torch.arange(x.shape[0]).to(x.device)


def ast_fbe_cnn(pretrained=True, pretrained_ckpt='pretrained_models/audioset_10_10_0.4593.pth',
                num_classes=4, **kwargs):
    """构造 FBE + CNN 二通道模型"""
    model = ASTFBECNN(
        label_dim=num_classes,
        input_fdim=128,
        input_tdim=798,  # 正确的输入尺寸
        **kwargs
    )

    # 加载预训练权重（FBE 部分）
    if pretrained:
        sd = model.state_dict()
        new_sd = {k: v for k, v in torch.load(pretrained_ckpt, map_location='cpu').items()
                  if k in sd and sd[k].shape == v.shape}
        sd.update(new_sd)
        model.load_state_dict(sd)
        print(f"加载预训练权重: {len(new_sd)} 层")

    return model
