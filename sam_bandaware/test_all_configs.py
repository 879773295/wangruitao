#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试所有消融配置是否能正常运行
只验证模型能加载到 GPU，前向传播不报错
"""

import torch
import os
os.environ['HF_HUB_OFFLINE'] = '1'

from src.model_bandaware import build_model

# 测试配置列表
configs = [
    {"name": "baseline", "use_same_band_mix": False, "use_band_importance": False, "use_freq_weighted_lam": False, "use_bandcl": False},
    {"name": "+Same-band", "use_same_band_mix": True, "use_band_importance": False, "use_freq_weighted_lam": False, "use_bandcl": False},
    {"name": "+BandImport", "use_same_band_mix": True, "use_band_importance": True, "use_freq_weighted_lam": False, "use_bandcl": False},
    {"name": "+FreqWeight", "use_same_band_mix": True, "use_band_importance": True, "use_freq_weighted_lam": True, "use_bandcl": False},
    {"name": "full", "use_same_band_mix": True, "use_band_importance": True, "use_freq_weighted_lam": True, "use_bandcl": True},
]

print("=== 快速测试所有消融配置 ===")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# 模拟输入（频谱图格式，和 dataset 输出一致）
batch_size = 2
x = torch.randn(batch_size, 1024, 128).cuda()  # AST 期望 [B, time, freq]
labels = torch.randint(0, 4, (batch_size,)).cuda()

for config in configs:
    print(f"\n=== 测试配置: {config['name']} ===")
    try:
        # 构建模型
        model = build_model(
            num_classes=4,
            num_bands=4,
            use_same_band_mix=config['use_same_band_mix'],
            use_band_importance=config['use_band_importance'],
            use_freq_weighted_lam=config['use_freq_weighted_lam'],
            use_bandcl=config['use_bandcl']
        ).cuda()

        # 测试 forward（无 patch_mix）
        model.eval()
        with torch.no_grad():
            logits, band_features, band_importance = model(x)
        print(f"  [OK] forward 无 patch_mix: logits={logits.shape}, band_features={band_features.shape}")

        # 测试 forward（有 patch_mix）
        model.train()
        with torch.no_grad():
            logits_mix, band_src, band_mix, band_imp, index = model(x, patch_mix=True, mix_lam=0.5)
        print(f"  [OK] forward 有 patch_mix: logits={logits_mix.shape}")

        # 测试 BandCL Loss
        if config['use_bandcl']:
            bandcl_loss = model.compute_bandcl_loss(band_src, band_mix)
            print(f"  [OK] BandCL Loss: {bandcl_loss.item():.4f}")

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [OK] 参数量: {total_params:.2f}M")

        print(f"  [OK] 配置 {config['name']} 测试通过！")

    except Exception as e:
        print(f"  [FAIL] 配置 {config['name']} 测试失败: {e}")
        import traceback
        traceback.print_exc()

print("\n=== 测试完成 ===")
