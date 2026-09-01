"""
Focal Loss (Lin et al., 2017) —— 用于呼吸音分类的长尾/类不平衡场景

针对 ICBHI 中 minority 类（wheeze / both）在训练时被 majority 类（normal）
主导梯度、导致完全学不到的问题。Focal Loss 通过 (1 - p_t)^gamma 自动降低
已分类良好（easy）样本的损失权重，把梯度让给难分样本（尤其是少数异常类）。

用法（main.py）：
    --loss focal --focal_gamma 2.0

配合 --weighted_sampler（逆频率重采样）一起用效果最好。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        # alpha: 可选类别权重（与 CrossEntropyLoss 的 weight 语义一致）
        # 用 register_buffer 以便 module.cuda() 时自动搬到 GPU
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)  # 正确类别的预测概率
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
