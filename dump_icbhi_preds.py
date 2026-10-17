"""
导出 ICBHI 测试集预测结果 → preds.npz（y_true + y_pred）
兼容两种 checkpoint 格式，自动识别：

  A) main.py 格式（原版 Patch-Mix）：{'args', 'model', 'classifier', ...}
     → 走 main.py 的 set_loader/set_model 流水线
  B) 裸 state_dict（sam_bandaware/train.py 等保存的 best_model.pth）
     → 走 sam_bandaware 的 ASTDataset + build_model 流水线
     → 消融开关必须用命令行参数传，与训练时保持一致！

用法：
    # A) main.py 格式的 checkpoint
    python dump_icbhi_preds.py --ckpt save/ast/best.pth --out preds_icbhi.npz

    # B) sam_bandaware full 配置（train.py 默认全开）
    python dump_icbhi_preds.py --ckpt sam_bandaware/checkpoints/full/best_model.pth \
        --out preds_full.npz

    # B) sam_bandaware baseline（四个模块全关）
    python dump_icbhi_preds.py --ckpt sam_bandaware/checkpoints/ablation_baseline/best_model.pth \
        --out preds_baseline.npz \
        --no_same_band_mix --no_band_importance --no_freq_weighted_lam --no_bandcl

然后画混淆矩阵：
    python plot_confusion.py --preds preds_icbhi.npz --labels Normal Crackle Wheeze Both

说明：
- A 格式：test_fold / 模型结构等自动从 ckpt['args'] 还原，可用 --test-fold 覆盖
- B 格式：state_dict 里没有训练参数，模型结构靠命令行开关重建，
          开关传错会报 missing/unexpected keys —— 按文件夹名对应回配置
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SAM_DIR = os.path.join(ROOT, 'sam_bandaware')


def binary_score(y_true, y_pred):
    """双口径：sp 两者相同(正常类召回)；
    se2c = 异常类被判成任一异常(2-class 折叠)；
    se4c = 异常类精确命中(Patch-Mix Eq.1)
    返回 (sp, se2c, score2c, acc, se4c, score4c)，单位 %"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sp = np.mean(y_pred[y_true == 0] == 0) * 100 if (y_true == 0).any() else 0.0
    ab = y_true > 0
    se2 = np.mean(y_pred[ab] > 0) * 100 if ab.any() else 0.0
    se4 = np.mean(y_pred[ab] == y_true[ab]) * 100 if ab.any() else 0.0
    acc = np.mean(y_true == y_pred) * 100
    return sp, se2, (sp + se2) / 2, acc, se4, (sp + se4) / 2


def guess_format(ckpt):
    """识别 checkpoint 格式：main.py 字典包 / 裸 state_dict"""
    if isinstance(ckpt, dict) and {'args', 'model'} <= set(ckpt.keys()):
        return 'main'
    return 'bare'


def strip_module_prefix(state_dict):
    """去掉 DataParallel 的 'module.' 前缀（保险用）"""
    keys = list(state_dict.keys())
    if keys and keys[0].startswith('module.'):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def dump_main(ckpt, args_in):
    """A) main.py 格式的 checkpoint：复用 main.py 的数据与模型逻辑"""
    from main import set_loader, set_model  # 惰性导入，只在需要时加载 main.py

    args = ckpt['args']
    if isinstance(args, dict):                 # 兼容以后存成 dict 的情况
        args = SimpleNamespace(**args)
    if args_in.test_fold is not None:
        args.test_fold = args_in.test_fold
    if args_in.batch_size is not None:
        args.batch_size = args_in.batch_size
    print(f'训练参数: model={args.model}, method={args.method}, test_fold={args.test_fold}, '
          f'n_cls={args.n_cls}, epoch={ckpt.get("epoch", "?")}')

    _, val_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)
    model.load_state_dict(ckpt['model'])
    classifier.load_state_dict(ckpt['classifier'])
    model.eval()
    classifier.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true, y_pred = [], []
    print('开始推理...')
    with torch.no_grad():
        for images, labels, metadata in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                features = model(images)
                output = classifier(features)
            preds = output.argmax(1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def dump_bare(ckpt, args_in):
    """B) 裸 state_dict（sam_bandaware 系列训练脚本保存的 best_model.pth）"""
    sys.path.insert(0, SAM_DIR)  # src.dataset / src.model_bandaware 在 sam_bandaware/ 下
    from torch.utils.data import DataLoader
    from transformers import ASTFeatureExtractor
    from src.dataset import ASTDataset
    from src.model_bandaware import build_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'数据: {args_in.data_path}')
    if not os.path.exists(args_in.data_path):
        raise FileNotFoundError(f'数据文件未找到: {args_in.data_path}')
    data = np.load(args_in.data_path)
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']

    local_model_path = os.path.join(ROOT, 'pretrained_models', 'ast')
    processor = ASTFeatureExtractor.from_pretrained(
        local_model_path if os.path.exists(local_model_path)
        else 'MIT/ast-finetuned-audioset-10-10-0.4593')

    batch_size = args_in.batch_size or 8
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=batch_size, shuffle=False)

    print('消融开关（必须与训练时一致）:')
    print(f'  use_same_band_mix:     {args_in.use_same_band_mix}')
    print(f'  use_band_importance:   {args_in.use_band_importance}')
    print(f'  use_freq_weighted_lam: {args_in.use_freq_weighted_lam}')
    print(f'  use_bandcl:            {args_in.use_bandcl}')
    print(f'  num_bands: {args_in.num_bands}, '
          f'hidden_ratio: {args_in.band_importance_hidden_ratio}, '
          f'temperature: {args_in.band_importance_temperature}, '
          f'bandcl_loss_type: {args_in.bandcl_loss_type}')

    model = build_model(
        num_classes=4,  # ICBHI 固定 4 类
        num_bands=args_in.num_bands,
        use_same_band_mix=args_in.use_same_band_mix,
        use_band_importance=args_in.use_band_importance,
        use_freq_weighted_lam=args_in.use_freq_weighted_lam,
        use_bandcl=args_in.use_bandcl,
        band_importance_hidden_ratio=args_in.band_importance_hidden_ratio,
        band_importance_temperature=args_in.band_importance_temperature,
        bandcl_loss_type=args_in.bandcl_loss_type,
    ).to(device)

    try:
        model.load_state_dict(strip_module_prefix(ckpt), strict=True)
    except RuntimeError as e:
        raise SystemExit(
            '加载权重失败：命令行消融开关与训练时不一致？\n'
            '（检查 --no_* / --use_* 是否与 checkpoint 所在文件夹名对应，\n'
            ' 例如 checkpoints/ablation_baseline → 四个模块全 --no_）\n'
            f'原始报错: {str(e)[:400]}')
    model.eval()

    y_true, y_pred = [], []
    print('开始推理...')
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            logits, _, _ = model(inputs, patch_mix=False)  # 推理与训练时验证一致
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description='导出 ICBHI 测试集预测（兼容两种 checkpoint 格式）')
    ap.add_argument('--ckpt', required=True, help='checkpoint 路径（main.py 字典包或裸 state_dict）')
    ap.add_argument('--out', default='preds_icbhi.npz', help='输出 npz')
    # main.py 格式专用
    ap.add_argument('--test-fold', default=None, help='[仅main格式] 覆盖 test_fold（official/0/1/2/3/4）')
    # 裸 state_dict 专用（与 sam_bandaware/train.py 对齐）
    ap.add_argument('--data-path',
                    default=os.path.join(SAM_DIR, 'icbhi_ast_16k_8s_metadata.npz'),
                    help='[仅裸权重格式] 预处理数据 npz 路径')
    ap.add_argument('--batch-size', type=int, default=None,
                    help='覆盖 batch_size（裸权重格式默认 8）')
    ap.add_argument('--num-bands', type=int, default=4)
    ap.add_argument('--use_same_band_mix', action='store_true', default=True)
    ap.add_argument('--no_same_band_mix', action='store_false', dest='use_same_band_mix')
    ap.add_argument('--use_band_importance', action='store_true', default=True)
    ap.add_argument('--no_band_importance', action='store_false', dest='use_band_importance')
    ap.add_argument('--use_freq_weighted_lam', action='store_true', default=True)
    ap.add_argument('--no_freq_weighted_lam', action='store_false', dest='use_freq_weighted_lam')
    ap.add_argument('--use_bandcl', action='store_true', default=True)
    ap.add_argument('--no_bandcl', action='store_false', dest='use_bandcl')
    ap.add_argument('--band_importance_hidden_ratio', type=int, default=4)
    ap.add_argument('--band_importance_temperature', type=float, default=1.0)
    ap.add_argument('--bandcl_loss_type', type=str, default='mse', choices=['mse', 'l1', 'cos'])
    args_in = ap.parse_args()

    print(f'加载 checkpoint: {args_in.ckpt}')
    ckpt = torch.load(args_in.ckpt, map_location='cpu')
    fmt = guess_format(ckpt)
    print(f'识别格式: {"main.py 字典包" if fmt == "main" else "裸 state_dict（sam_bandaware 系列）"}')

    if fmt == 'main':
        y_true, y_pred = dump_main(ckpt, args_in)
    else:
        y_true, y_pred = dump_bare(ckpt, args_in)

    np.savez(args_in.out, y_true=y_true, y_pred=y_pred)
    print(f'已保存: {args_in.out}  （{len(y_true)} 条测试样本）')

    n = max(int(y_true.max()), int(y_pred.max())) + 1
    print('\n混淆矩阵（行=真实标签）:')
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    for row in cm:
        print('  ' + ' '.join(f'{v:5d}' for v in row))

    sp, se2, sc2, acc, se4, sc4 = binary_score(y_true, y_pred)
    print(f'\n * S_p: {sp:.2f}, S_e(2c): {se2:.2f}, Score(2c): {sc2:.2f} | S_e(4c): {se4:.2f}, Score(4c): {sc4:.2f}, Acc: {acc:.2f}')


if __name__ == '__main__':
    main()
