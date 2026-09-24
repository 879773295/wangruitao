#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型融合：权重平均

把基线和 SupCon 的权重平均，得到一个融合模型。
结构相同的情况下，权重平均是简单有效的方法。
"""

import os
import torch
import copy
import main as M


def average_model_weights(model1_state_dict, model2_state_dict, alpha=0.5):
    """平均两个模型的权重"""
    fused_dict = copy.deepcopy(model1_state_dict)
    for key in fused_dict.keys():
        if key in model2_state_dict:
            fused_dict[key] = alpha * model1_state_dict[key] + (1 - alpha) * model2_state_dict[key]
    return fused_dict


def save_fused_model(fused_state_dict, fused_classifier_state_dict, args, save_path):
    """保存融合模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model': fused_state_dict,
        'classifier': fused_classifier_state_dict,
        'args': args,
        'epoch': 0,
        'alpha': 0.5
    }, save_path)
    print(f"融合模型已保存: {save_path}")


def main():
    # 1. 加载两个模型
    print("=== 加载模型 ===")

    # 基线
    print("加载基线模型...")
    ckpt_baseline = torch.load('./save/icbhi_ast_patchmix_cl_torch201_seed1/best.pth', map_location='cpu')
    model_baseline = ckpt_baseline['model']
    classifier_baseline = ckpt_baseline['classifier']
    args_baseline = ckpt_baseline['args']

    # SupCon
    print("加载 SupCon 模型...")
    ckpt_supcon = torch.load('./save/icbhi_ast_patchmix_cl_supcon_seed1_e10/best.pth', map_location='cpu')
    model_supcon = ckpt_supcon['model']
    classifier_supcon = ckpt_supcon['classifier']

    # 2. 平均权重（α=0.5，各占一半）
    print("\n=== 融合权重 (α=0.5) ===")
    fused_model = average_model_weights(model_baseline, model_supcon, alpha=0.5)
    fused_classifier = average_model_weights(classifier_baseline, classifier_supcon, alpha=0.5)

    # 3. 保存融合模型
    save_path = './save/fused_baseline_supcon_alpha0.5.pth'
    save_fused_model(fused_model, fused_classifier, args_baseline, save_path)

    # 4. 验证融合模型
    print("\n=== 验证融合模型 ===")
    # 加载并测试
    ckpt_fused = torch.load(save_path, map_location='cpu')
    print(f"融合模型 checkpoint: {save_path}")
    print(f"  alpha: {ckpt_fused.get('alpha')}")
    print(f"  args.model: {ckpt_fused['args'].model}")

    print("\n✅ 融合完成！下一步：")
    print(f"  python ensemble_eval.py --ckpts {save_path} --tta 0")


if __name__ == '__main__':
    main()
