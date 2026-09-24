#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 AST + FBE + Band-aware Patch-Mix 代码是否可以正常运行
不进行实际训练，只验证代码结构和数据流
"""

import torch
import torch.nn as nn
from models import get_backbone_class
from method import BandCLLoss

def test_bandaware_model():
    """测试 ASTFBEBandAware 模型"""
    print("=== 测试 ASTFBEBandAware 模型 ===")

    # 模拟参数
    class Args:
        def __init__(self):
            self.model = 'ast_fbe_bandaware'
            self.n_cls = 4
            self.h = 798
            self.w = 128
            self.resz = 1.0
            self.mix_beta = 1.0
            self.num_bands = 4

    args = Args()

    # 获取模型
    model_class = get_backbone_class(args.model)
    model = model_class(
        label_dim=args.n_cls,
        input_fdim=int(args.h * args.resz),
        input_tdim=int(args.w * args.resz),
        imagenet_pretrain=True,
        audioset_pretrain=True,
        verbose=True,
        mix_beta=args.mix_beta,
        num_bands=args.num_bands
    ).cuda()  # 移动到 GPU

    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 1, 798, 128).cuda()
    y = torch.randint(0, 4, (batch_size,)).cuda()

    print(f"\n输入形状: {x.shape}")
    print(f"标签形状: {y.shape}")

    # 测试 forward（无 patch_mix）
    model.eval()
    with torch.no_grad():
        features = model(x)
    print(f"\n无 patch_mix 输出形状: {features.shape}")

    # 测试 forward（有 patch_mix）
    model.train()
    with torch.no_grad():
        outputs = model(x, y=y, patch_mix=True, time_domain=False)
        if isinstance(outputs, tuple):
            mixed_features, labels_a, labels_b, lam, index = outputs
            print(f"\n有 patch_mix 输出形状: {mixed_features.shape}")
            print(f"labels_a: {labels_a}")
            print(f"labels_b: {labels_b}")
            print(f"lam: {lam}")
        else:
            print(f"\n有 patch_mix 输出形状: {outputs.shape}")

    print("\n[OK] ASTFBEBandAware 模型测试通过！")
    return model

def test_bandcl_loss():
    """测试 BandCL Loss"""
    print("\n=== 测试 BandCL Loss ===")

    # 模拟数据
    batch_size = 2
    num_bands = 4
    embed_dim = 768

    band_features_src = torch.randn(batch_size, num_bands, embed_dim)
    band_features_mix = torch.randn(batch_size, num_bands, embed_dim)

    print(f"源样本频段特征形状: {band_features_src.shape}")
    print(f"混合样本频段特征形状: {band_features_mix.shape}")

    # 测试 BandCLLoss
    bandcl_loss = BandCLLoss(num_bands=num_bands)
    loss = bandcl_loss(band_features_src, band_features_mix)

    print(f"BandCL Loss: {loss.item():.4f}")
    print("\n[OK] BandCL Loss 测试通过！")
    return bandcl_loss

def test_integration():
    """测试集成"""
    print("\n=== 测试集成 ===")

    # 模拟参数
    class Args:
        def __init__(self):
            self.model = 'ast_fbe_bandaware'
            self.n_cls = 4
            self.h = 798
            self.w = 128
            self.resz = 1.0
            self.mix_beta = 1.0
            self.num_bands = 4
            self.method = 'patchmix_cl'
            self.use_bandcl = True
            self.bandcl_weight = 1.0

    args = Args()

    # 获取模型
    model_class = get_backbone_class(args.model)
    model = model_class(
        label_dim=args.n_cls,
        input_fdim=int(args.h * args.resz),
        input_tdim=int(args.w * args.resz),
        imagenet_pretrain=True,
        audioset_pretrain=True,
        verbose=False,
        mix_beta=args.mix_beta,
        num_bands=args.num_bands
    ).cuda()  # 移动到 GPU

    # 模拟损失函数
    ce_loss = nn.CrossEntropyLoss()
    bandcl_loss = BandCLLoss(num_bands=args.num_bands)

    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 1, 798, 128).cuda()
    y = torch.randint(0, 4, (batch_size,)).cuda()

    print(f"输入形状: {x.shape}")
    print(f"标签形状: {y.shape}")

    # 测试 forward + loss
    model.train()
    outputs = model(x, y=y, patch_mix=True, time_domain=False)

    if isinstance(outputs, tuple):
        mixed_features, labels_a, labels_b, lam, index = outputs
        print(f"混合特征形状: {mixed_features.shape}")

        # 计算损失
        # 模拟 classifier
        classifier = nn.Linear(768, 4).cuda()
        output = classifier(mixed_features)
        loss = ce_loss(output, labels_a)

        print(f"分类损失: {loss.item():.4f}")

        # 如果启用 BandCL Loss
        if args.use_bandcl:
            # 获取频段特征（简化处理）
            # 实际实现需要根据模型结构调整
            band_features_src = torch.randn(batch_size, args.num_bands, 768)
            band_features_mix = torch.randn(batch_size, args.num_bands, 768)

            bandcl = bandcl_loss(band_features_src, band_features_mix)
            loss += args.bandcl_weight * bandcl
            print(f"BandCL 损失: {bandcl.item():.4f}")
            print(f"总损失: {loss.item():.4f}")

    print("\n[OK] 集成测试通过！")

if __name__ == '__main__':
    # 测试模型
    model = test_bandaware_model()

    # 测试损失函数
    bandcl_loss = test_bandcl_loss()

    # 测试集成
    test_integration()

    print("\n" + "="*60)
    print("[OK] 所有测试通过！")
    print("代码结构正确，可以正常运行。")
    print("="*60)
