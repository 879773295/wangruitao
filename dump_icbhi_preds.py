"""
导出 ICBHI 官方测试集的预测结果 → preds.npz（y_true + y_pred）

用法：
    python dump_icbhi_preds.py --ckpt save/ast/best.pth --out preds_icbhi.npz

然后画混淆矩阵：
    python plot_confusion.py --preds preds_icbhi.npz --labels Normal Crackle Wheeze Both

说明：
- checkpoint 里的 args 是训练时保存的，test_fold / 模型结构等自动还原
- 可用 --test-fold 覆盖划分（official / 0 / 1 / 2 / 3 / 4）
"""
import argparse
import os

import numpy as np
import torch

from main import set_loader, set_model


def binary_score(y_true, y_pred):
    """ICBHI 二分类口径：sp = 正常类召回, se = 异常类被预测为异常的比例"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(y_true.max()) + 1
    sp = np.mean(y_pred[y_true == 0] == 0) * 100 if (y_true == 0).any() else 0.0
    ab = y_true > 0
    se = np.mean(y_pred[ab] > 0) * 100 if ab.any() else 0.0
    acc = np.mean(y_true == y_pred) * 100
    return sp, se, (sp + se) / 2, acc


def main():
    ap = argparse.ArgumentParser(description='导出 ICBHI 测试集预测')
    ap.add_argument('--ckpt', required=True, help='best.pth 路径')
    ap.add_argument('--out', default='preds_icbhi.npz', help='输出 npz')
    ap.add_argument('--test-fold', default=None, help='覆盖 test_fold（official/0/1/2/3/4）')
    ap.add_argument('--batch-size', type=int, default=None, help='覆盖 batch_size')
    args_in = ap.parse_args()

    print(f'加载 checkpoint: {args_in.ckpt}')
    ckpt = torch.load(args_in.ckpt, map_location='cpu')
    args = ckpt['args']  # 训练时保存的完整 args
    if args_in.test_fold is not None:
        args.test_fold = args_in.test_fold
    if args_in.batch_size is not None:
        args.batch_size = args_in.batch_size
    print(f'训练参数: model={args.model}, method={args.method}, test_fold={args.test_fold}, '
          f'n_cls={args.n_cls}, epoch={ckpt.get("epoch", "?")}')

    # 构建数据与模型（复用 main.py 的逻辑，保证与训练/验证完全一致）
    _, val_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)
    model.load_state_dict(ckpt['model'])
    classifier.load_state_dict(ckpt['classifier'])
    model.eval()
    classifier.eval()

    y_true, y_pred = [], []
    print('开始推理...')
    with torch.no_grad():
        for images, labels, metadata in val_loader:
            images = images.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                features = model(images)
                output = classifier(features)
            preds = output.argmax(1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    np.savez(args_in.out, y_true=y_true, y_pred=y_pred)
    print(f'已保存: {args_in.out}  （{len(y_true)} 条测试样本）')

    n = int(args.n_cls)
    print('\n混淆矩阵（行=真实标签）:')
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    for row in cm:
        print('  ' + ' '.join(f'{v:5d}' for v in row))

    sp, se, sc, acc = binary_score(y_true, y_pred)
    print(f'\n * S_p: {sp:.2f}, S_e: {se:.2f}, Score: {sc:.2f}, Acc: {acc:.2f}')


if __name__ == '__main__':
    main()
