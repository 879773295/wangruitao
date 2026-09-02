#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合 Loss：InfoNCE + SupCon

同时使用两种对比学习目标：
- InfoNCE：优化样本级不变性（对异常敏感）
- SupCon：优化类别级可分性（对正常聚焦）

融合后模型可以同时学到两种特征表示的优势。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedContrastiveLoss(nn.Module):
    """
    融合对比损失：InfoNCE + SupCon

    Args:
        temperature: 温度参数
        alpha: InfoNCE 权重
        beta: SupCon 权重
    """
    def __init__(self, temperature=0.06, alpha=1.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, proj1, proj2, labels, labels_b=None, lam=None, index=None, args=None):
        """
        Args:
            proj1: [B, D] 原图特征
            proj2: [B, D] 混patch后的特征
            labels: [B] 原图标签
            labels_b: [B] 混patch后的标签（InfoNCE 用）
            lam: 混合比例（InfoNCE 用）
            index: 伙伴索引（InfoNCE 用）
            args: 参数
        Returns:
            loss: 标量
        """
        batch_size = proj1.shape[0]

        # 1. InfoNCE Loss（样本级对比）
        infonce_loss = self._infonce_loss(proj1, proj2, labels, labels_b, lam, index, args)

        # 2. SupCon Loss（类别级对比）
        supcon_loss = self._supcon_loss(proj1, labels)

        # 3. 融合
        loss = self.alpha * infonce_loss + self.beta * supcon_loss

        return loss

    def _infonce_loss(self, proj1, proj2, labels, labels_b, lam, index, args):
        """InfoNCE Loss（原始 PatchMixConLoss 的逻辑）"""
        proj1, proj2 = F.normalize(proj1), F.normalize(proj2)

        # 相似度矩阵
        anchor_dot_contrast = torch.div(torch.matmul(proj2, proj1.T), self.temperature)

        # 正样本掩码：自己 + 伙伴
        mask_a = torch.eye(proj1.shape[0]).cuda()
        mask_b = torch.zeros(proj1.shape[0], proj1.shape[0]).cuda()
        mask_b[torch.arange(proj1.shape[0]).unsqueeze(1), index.view(-1, 1)] = 1

        mask = lam * mask_a + (1 - lam) * mask_b

        # 数值稳定
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)

        # 如果指定只用异类负样本
        if hasattr(args, 'negative_pair') and args.negative_pair == 'diff_label':
            labels_a = labels.contiguous().view(-1, 1)
            logits_mask = torch.ne(labels_a, labels_a.T).cuda() + (mask_a.bool() + mask_b.bool())
            exp_logits = exp_logits * logits_mask.float()

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()

        return loss

    def _supcon_loss(self, features, labels):
        """SupCon Loss（直接用类别标签）"""
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        # 相似度矩阵
        sim = torch.matmul(features, features.T) / self.temperature

        # 同类样本掩码
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()

        # 排除自身
        mask_neg = 1 - torch.eye(batch_size, device=features.device)

        # 计算 loss
        exp_sim = torch.exp(sim) * mask_neg
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss


# 为了兼容性，也导出单独的 InfoNCE 和 SupCon
from .patchmix_cl import PatchMixConLoss as InfoNCE
from .supcon import PatchMixSupConLoss as SupCon
