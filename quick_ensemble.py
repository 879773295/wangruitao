#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速加权集成：基线 + SupCon

直接用已有 ensemble_eval.py 的逻辑，但改成加权平均。
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import main as M
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score, get_score_both
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    # 1. 加载数据
    print("=== 加载数据 ===")
    # 用 ensemble_eval.py 的方式加载
    ckpt0 = torch.load('./save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth', map_location='cpu')
    targs0 = ckpt0['args']
    train_loader, val_loader, targs0 = M.set_loader(targs0)

    # 2. 加载模型并提取 logits
    print("\n=== 加载模型 ===")
    device = 'cuda'

    # 基线
    print("加载基线模型...")
    ckpt_baseline = torch.load('./save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth', map_location='cpu')
    model_baseline, classifier_baseline, _, _, _ = M.set_model(ckpt_baseline['args'])
    model_baseline.load_state_dict(ckpt_baseline['model'])
    classifier_baseline.load_state_dict(ckpt_baseline['classifier'])
    model_baseline.cuda().eval()
    classifier_baseline.cuda().eval()

    # SupCon
    print("加载 SupCon 模型...")
    ckpt_supcon = torch.load('./save/icbhi_ast_patchmix_cl_supcon_seed1_e10/best.pth', map_location='cpu')
    model_supcon, classifier_supcon, _, _, _ = M.set_model(ckpt_supcon['args'])
    model_supcon.load_state_dict(ckpt_supcon['model'])
    classifier_supcon.load_state_dict(ckpt_supcon['classifier'])
    model_supcon.cuda().eval()
    classifier_supcon.cuda().eval()

    # 3. 提取 logits
    print("\n=== 提取 logits ===")
    all_logits_baseline = []
    all_logits_supcon = []
    all_labels = []

    with torch.no_grad():
        for images, labels, metadata in val_loader:
            images = images.cuda()
            with torch.cuda.amp.autocast():
                feat_b = model_baseline(images)
                logit_b = classifier_baseline(feat_b)

                feat_s = model_supcon(images)
                logit_s = classifier_supcon(feat_s)

            all_logits_baseline.append(logit_b.cpu())
            all_logits_supcon.append(logit_s.cpu())
            all_labels.append(labels)

    logits_baseline = torch.cat(all_logits_baseline, 0)
    logits_supcon = torch.cat(all_logits_supcon, 0)
    labels = torch.cat(all_labels, 0)

    print(f"Logits shape: {logits_baseline.shape}")

    # 4. 搜索最优权重
    print("\n=== 搜索最优权重 ===")
    best_score = 0
    best_w = None

    for w in np.arange(0, 1.05, 0.05):
        # 加权平均
        logits = w * logits_baseline + (1 - w) * logits_supcon
        preds = torch.argmax(logits, dim=1)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.asarray(labels)
        # 双口径：best 仍按 4c（历史口径），2c 同行显示
        sp, se4, se2, sc4, sc2 = get_score_both(preds_np, labels_np)
        se, sc = se4, sc4

        if sc > best_score:
            best_score = sc
            best_sp = sp
            best_se = se
            best_w = w
            print(f"  w_baseline={w:.2f}, w_supcon={1-w:.2f} -> S_p={sp:.2f}, Score(4c)={sc:.2f} | Score(2c)={sc2:.2f} ★")

    # 5. 打印结果
    print(f"\n{'='*60}")
    print(f"最优权重: w_baseline={best_w:.2f}, w_supcon={1-best_w:.2f}")
    print(f"Score(4c): {best_score:.2f}")
    print(f"S_p: {best_sp:.2f}, S_e(4c): {best_se:.2f}")
    print(f"{'='*60}")

    # 对比
    print(f"\n对比:")
    print(f"  基线单模型: Score=60.35, S_p=76.19, S_e=44.52")
    print(f"  SupCon单模型: Score=58.66, S_p=71.18, S_e=46.13")
    print(f"  **加权融合: Score={best_score:.2f}, S_p={best_sp:.2f}, S_e={best_se:.2f}**")


if __name__ == '__main__':
    main()
