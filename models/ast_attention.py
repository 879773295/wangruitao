"""
AST + Attention Pooling

替换平均池化，让模型自己学"关注哪些时间步"。
异常样本可能只占很短的时间段（比如只有一小段啰音），
Attention Pooling 能让模型聚焦在异常段，而不是被正常段稀释。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .ast import ASTModel


class AttentionPooling(nn.Module):
    """Attention Pooling: 让模型自己学关注哪些时间步"""
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, N, D] (B=batch, N=num_patches+cls+distill, D=embed_dim)
        return: [B, D] (pooled features)
        """
        # Self-attention
        attn_out, attn_weights = self.attn(x, x, x)  # [B, N, D]
        x = self.norm(x + attn_out)  # residual connection

        # 取 cls token（第一个 token）作为输出
        return x[:, 0, :]  # [B, D]


class ASTModelAttention(nn.Module):
    """
    AST + Attention Pooling

    与原始 AST 的区别：
    - 原始 AST：用 cls token 或 average pooling
    - AST+Attention：用 self-attention 让模型自己关注哪些 patch

    使用方式：
        --model ast_attention
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, audioset_pretrain=False,
                 model_size='base384', verbose=True, mix_beta=None,
                 num_attention_heads=8):
        super().__init__()

        # 复用原始 AST 的初始化逻辑
        self.ast = ASTModel(
            label_dim=label_dim,
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
            verbose=verbose,
            mix_beta=mix_beta
        )

        # 替换池化层
        self.final_feat_dim = self.ast.final_feat_dim
        self.attn_pool = AttentionPooling(
            embed_dim=self.final_feat_dim,
            num_heads=num_attention_heads
        )

        # 保留 mlp_head（用于分类）
        self.mlp_head = self.ast.mlp_head

        if verbose:
            print(f"  Attention Pooling: num_heads={num_attention_heads}")

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        与原始 AST 相同的接口
        """
        # 获取 AST 的中间特征（不经过 mlp_head）
        # 需要修改 AST 的 forward 来返回特征

        # 方法：直接调用 ast 的 forward，但 patch_mix 时需要特殊处理
        if patch_mix:
            # PatchMix: 先获取特征，再混 patch
            features = self.ast.forward_features(x)  # [B, N, D]

            # 做 patch mix
            mix_features, labels_a, labels_b, lam, index = self.ast.patch_mix(
                features, y=y, time_domain=time_domain
            )

            # Attention Pooling
            pooled = self.attn_pool(mix_features)  # [B, D]

            return pooled, labels_a, labels_b, lam, index
        else:
            # 普通前向：获取特征 -> Attention Pooling
            features = self.ast.forward_features(x)  # [B, N, D]
            pooled = self.attn_pool(features)  # [B, D]

            return pooled

    def forward_features(self, x):
        """获取特征（不经过池化）"""
        return self.ast.forward_features(x)


# 为了兼容性，也导出原始 ASTModel
from .ast import ASTModel as ASTModelBase
