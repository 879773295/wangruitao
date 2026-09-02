import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon, 2020)
    https://arxiv.org/abs/1911.09666

    最大化同类样本相似度，最小化异类样本相似度。
    比 InfoNCE 更直接——直接用类别信息构建正负样本对。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] 特征向量
            labels: [B] 类别标签
        Returns:
            loss: 标量
        """
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        # 计算相似度矩阵 [B, B]
        sim = torch.matmul(features, features.T) / self.temperature

        # 构建同类样本掩码 [B, B]
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()  # 同类=1, 异类=0

        # 排除自身（对角线）
        mask_neg = 1 - torch.eye(batch_size, device=features.device)

        # 计算 loss
        exp_sim = torch.exp(sim) * mask_neg  # 去掉自身
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 只对正样本对计算 loss
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos
        return loss.mean()


class PatchMixSupConLoss(nn.Module):
    """
    PatchMix + Supervised Contrastive Loss
    结合 PatchMix 的数据增强 + SupCon 的监督对比学习

    训练流程：
    1. 原图特征 proj1 = projector(backbone(x))   ← 用于分类
    2. 混 patch 后的特征 proj2 = projector(backbone(mix_x))
    3. 分类损失：CE(proj1, labels)
    4. 对比损失：SupCon(proj1, labels) + SupCon(proj2, labels)
    5. 对比增强：把原图和混 patch 后的图都作为正样本对
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2, labels):
        """
        Args:
            features1: [B, D] 原图特征
            features2: [B, D] 混 patch 后的特征
            labels: [B] 类别标签
        Returns:
            loss: 标量
        """
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        batch_size = features1.shape[0]

        # 拼接两个视图的特征 [2B, D]
        features = torch.cat([features1, features2], dim=0)
        labels_all = torch.cat([labels, labels], dim=0)

        # 计算相似度矩阵 [2B, 2B]
        sim = torch.matmul(features, features.T) / self.temperature

        # 构建同类样本掩码 [2B, 2B]
        labels_all = labels_all.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels_all, labels_all.T).float()

        # 排除自身
        mask_neg = 1 - torch.eye(2 * batch_size, device=features.device)

        # 计算 loss
        exp_sim = torch.exp(sim) * mask_neg
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos
        return loss.mean()
