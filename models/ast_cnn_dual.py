"""
AST + CNN 双通道模型
==========================
创新点：在原有 AST（Audio Spectrogram Transformer）通道基础上，
       增加一个 CNN 分支作为第二通道，提取局部频谱特征，
       两条通道特征拼接融合后进行分类。

通道设计：
  - 通道1（AST）：Patch Embedding → Transformer Encoder → CLS token
  - 通道2（CNN）：Conv Block → 全局池化 → 特征向量
  - 融合：Concat(ast_feat, cnn_feat) → Linear → 分类

新建文件，不覆盖原有 ast.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.cuda.amp import autocast
from models.ast import ASTModel


class CNNBranch(nn.Module):
    """
    CNN 分支：从 mel 频谱图中提取局部时频特征
    输入: (B, 1, n_mels, time_frames)
    输出: (B, cnn_embed_dim)
    """

    def __init__(self, cnn_embed_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B,1,128,T) -> (B,32,64,T/2)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (B,32,64,T/2) -> (B,64,32,T/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (B,64,32,T/4) -> (B,128,16,T/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: (B,128,16,T/8) -> (B,cnn_embed_dim,8,T/16)
            nn.Conv2d(128, cnn_embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B, cnn_embed_dim, 1, 1)
        )

    def forward(self, x):
        # x: (B, 1, n_mels, time_frames)
        x = self.features(x)   # (B, cnn_embed_dim, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, cnn_embed_dim)
        return x


class ASTModelDualCNN(ASTModel):
    """
    AST + CNN 双通道模型

    继承 ASTModel，新增 CNN 分支。
    forward 时同时跑 AST 和 CNN 两条通道，拼接融合后分类。
    """

    def __init__(self, *args, cnn_embed_dim=256, label_dim=4, **kwargs):
        super().__init__(*args, label_dim=label_dim, **kwargs)
        self.cnn_embed_dim = cnn_embed_dim

        # CNN 分支
        self.cnn_branch = CNNBranch(cnn_embed_dim=cnn_embed_dim)

        # 原始 AST 分类头的输入维度（ViT-base hidden dim = 768）
        ast_dim = self.original_embedding_dim  # 768

        # 融合分类头：[AST cls | CNN feat] -> num_classes
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(ast_dim + cnn_embed_dim),
            nn.Linear(ast_dim + cnn_embed_dim, label_dim)
        )

        # 更新特征维度（供 Projector 使用）
        self.final_feat_dim = ast_dim + cnn_embed_dim

    @torch.no_grad()
    def _extract_cnn_input(self, x_raw):
        """
        从原始输入提取 CNN 需要的 mel 频谱
        x_raw: (B, T, F) 或 (B, 1, T, F)
        返回: (B, 1, F, T)  适合 Conv2d
        """
        if x_raw.dim() == 4:
            x_raw = x_raw.squeeze(1)  # (B,1,T,F) -> (B,T,F)
        # x_raw: (B, T, F) -> (B, 1, F, T)
        x_cnn = x_raw.transpose(1, 2).unsqueeze(1)
        return x_cnn

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        双通道前向传播
        :param x: (batch_size, time_frame_num, frequency_bins)
        :return: 分类 logits（非 patch_mix）或 (logits, y_a, y_b, lam, index)
        """
        # ===== 通道2：CNN =====
        x_cnn = self._extract_cnn_input(x)   # (B,1,F,T)
        cnn_feat = self.cnn_branch(x_cnn)    # (B, cnn_embed_dim)

        # ===== 通道1：AST（复用父类逻辑）=====
        x = x.transpose(2, 3)
        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1

        B = x.shape[0]
        x = self.v.patch_embed(x)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        ast_feat = (x[:, 0] + x[:, 1]) / 2  # (B, 768)

        # ===== 融合 =====
        fused = torch.cat([ast_feat, cnn_feat], dim=1)  # (B, 768+cnn_embed_dim)

        if not patch_mix:
            return fused  # 返回特征，由外部 classifier 做分类
        else:
            return fused, y_a, y_b, lam, index


def ast_cnn_dual(pretrained=True, pretrained_ckpt='pretrained_models/audioset_10_10_0.4593.pth',
                 num_classes=4, **kwargs):
    """构造 AST + CNN 双通道模型"""
    input_fdim = 128
    input_tdim = 1024
    model = ASTModelDualCNN(
        label_dim=num_classes,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        imagenet_pretrain=pretrained,
        audioset_pretrain=pretrained,
        cnn_embed_dim=256,
        **kwargs
    )
    if pretrained:
        sd = model.state_dict()
        new_sd = {k: v for k, v in torch.load(pretrained_ckpt, map_location='cpu').items()
                  if k in sd and sd[k].shape == v.shape}
        sd.update(new_sd)
        model.load_state_dict(sd)
    return model
