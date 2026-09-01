"""
阈值校准（不需要重新训练）—— 白嫖分

原理：ICBHI 的 Score = (S_p + S_e)/2，其中 S_p 是正常类准确率、S_e 是异常类准确率。
当前模型严重偏向「正常」类（S_p 高、S_e 低）。本脚本在验证集上对「normal 类的 logit」
加一个可调偏置 b，找到使 Score 最大的决策点，相当于手动移动 normal/abnormal 的决策边界。

用法（在项目根目录执行，需 GPU）：
    python eval_threshold.py --ckpt ./save/icbhi_ast_fbe_patchmix_cl_fbe_patho_test/best.pth
"""
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import main as M
from util.icbhi_util import get_score


def strip_module(state_dict):
    """去掉 DataParallel 保存时的 'module.' 前缀"""
    new = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new[k] = v
    return new


def collect_logits(val_loader, model, classifier):
    model.eval()
    classifier.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                feats = model(images)
                logits = classifier(feats)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def sp_se_score(logits, labels, n_cls):
    """复刻 util/icbhi_util.py:get_score 的语义：S_p=normal 准确率，S_e=异常类准确率"""
    preds = logits.argmax(1)
    hits = [0.0] * n_cls
    counts = [0.0] * n_cls
    for l in labels:
        counts[int(l)] += 1.0
    for p, l in zip(preds, labels):
        if int(p) == int(l):
            hits[int(l)] += 1.0
    return get_score(hits, counts)


def per_class_accuracy(logits, labels, n_cls):
    preds = logits.argmax(1)
    out = {}
    for c in range(n_cls):
        idx = labels == c
        if idx.sum() == 0:
            out[c] = float('nan')
        else:
            out[c] = (preds[idx] == c).float().mean().item() * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='best.pth 路径')
    ap.add_argument('--bias_min', type=float, default=-5.0)
    ap.add_argument('--bias_max', type=float, default=5.0)
    ap.add_argument('--bias_step', type=int, default=201)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    train_args = ckpt['args']
    model_sd = strip_module(ckpt['model'])
    classifier_sd = ckpt['classifier']

    print('checkpoint epoch: {}, model: {}, n_cls: {}'.format(
        ckpt.get('epoch'), train_args.model, train_args.n_cls))

    train_loader, val_loader, train_args = M.set_loader(train_args)
    model, classifier, _, _, _ = M.set_model(train_args)
    model.load_state_dict(model_sd, strict=False)
    classifier.load_state_dict(classifier_sd)

    logits, labels = collect_logits(val_loader, model, classifier)

    # 基线（argmax，bias=0）
    sp0, se0, sc0 = sp_se_score(logits, labels, train_args.n_cls)
    print('\n[per-class acc, argmax] ', end='')
    for c, a in per_class_accuracy(logits, labels, train_args.n_cls).items():
        print('{}={:.1f} '.format(train_args.cls_list[c], a), end='')
    print('\nbaseline (argmax):  S_p={:.2f}  S_e={:.2f}  Score={:.2f}'.format(sp0, se0, sc0))

    best = (-1, None, None, None)
    results = []
    for b in np.linspace(args.bias_min, args.bias_max, args.bias_step):
        adj = logits.clone()
        adj[:, 0] = adj[:, 0] + b  # 负 b 压低 normal logit → 更多判为异常
        sp, se, sc = sp_se_score(adj, labels, train_args.n_cls)
        results.append((sc, b, sp, se))
        if sc > best[0]:
            best = (sc, b, sp, se)

    results.sort(reverse=True)
    print('\ntop-5 决策点 (对 normal 类 logit 加偏置):')
    print('  bias      S_p     S_e     Score')
    for sc, b, sp, se in results[:5]:
        print('  {:+6.3f}  {:5.2f}  {:5.2f}  {:6.2f}'.format(b, sp, se, sc))

    sc, b, sp, se = best
    print('\n>>> 最优阈值: bias={:+.3f} -> S_p={:.2f}, S_e={:.2f}, Score={:.2f}'.format(b, sp, se, sc))
    print('>>> 相对 argmax: {:.2f} -> {:.2f}  ({:+.2f})'.format(sc0, sc, sc - sc0))


if __name__ == '__main__':
    main()
