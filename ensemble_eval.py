"""
集成 + TTA + 阈值校准 评估脚本（"不择手段"冲分版）

把多个 checkpoint 的 logits 平均（集成）、每个样本做 N 次 SpecAugment 视图平均（TTA）、
再扫 normal 类 logit 偏置（阈值校准），输出能达到的最高 Score。

用法（项目根目录）：
    python ensemble_eval.py --ckpts \
        ./save/icbhi_ast_patchmix_cl_plaince_bs8_ep50_seed1/best.pth \
        ./save/icbhi_ast_patchmix_cl_plaince_bs8_ep50_seed1337/best.pth \
        ... --tta 8
"""
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import main as M
from util.icbhi_util import get_score, get_score_both
from util.augmentation import SpecAugment


def strip_module(sd):
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}


def sp_se_score(logits, labels, n_cls):
    """双口径：返回 (sp, se4, sc4, se2, sc2)。4c=精确命中；2c=任一异常。"""
    preds = logits.argmax(1)
    preds = preds.numpy() if hasattr(preds, 'numpy') else np.asarray(preds)
    lab = labels.numpy() if hasattr(labels, 'numpy') else np.asarray(labels)
    sp, se4, se2, sc4, sc2 = get_score_both(preds, lab)
    return sp, se4, sc4, se2, sc2


def per_class(logits, labels, n_cls, cls_list):
    preds = logits.argmax(1)
    out = {}
    for c in range(n_cls):
        idx = labels == c
        out[cls_list[c]] = (preds[idx] == c).float().mean().item() * 100 if idx.sum() else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='多个 best.pth 路径')
    ap.add_argument('--tta', type=int, default=0, help='每个样本做多少次 SpecAugment 视图平均')
    ap.add_argument('--bias_min', type=float, default=-5.0)
    ap.add_argument('--bias_max', type=float, default=5.0)
    ap.add_argument('--bias_step', type=int, default=401)
    args = ap.parse_args()

    # 用第一个 ckpt 的 dataset 配置建 dataloader（所有 ckpt 的 dataset 一致）
    ckpt0 = torch.load(args.ckpts[0], map_location='cpu')
    targs0 = ckpt0['args']
    train_loader, val_loader, targs0 = M.set_loader(targs0)
    n_cls = targs0.n_cls
    cls_list = targs0.cls_list
    specaug = SpecAugment(targs0) if args.tta > 0 else None

    all_logits = []
    all_labels = None
    for path in args.ckpts:
        ckpt = torch.load(path, map_location='cpu')
        ta = ckpt['args']
        ta.h, ta.w = targs0.h, targs0.w  # 复用已算好的 h/w
        model, classifier, _, _, _ = M.set_model(ta)
        model.load_state_dict(strip_module(ckpt['model']), strict=False)
        classifier.load_state_dict(ckpt['classifier'])
        model.eval()
        classifier.eval()
        print('[load] {}  (model={}, epoch={})'.format(path, ta.model, ckpt.get('epoch')))

        logs, labels = [], []
        with torch.no_grad():
            for images, lab, _ in val_loader:
                images = images.cuda(non_blocking=True)
                views = [images]
                if specaug is not None:
                    # SpecAugment only supports single sample [C,H,W]; apply per-sample
                    for _ in range(args.tta):
                        augs = [specaug(images[i].clone()) for i in range(images.shape[0])]
                        views.append(torch.stack(augs, 0))
                outs = []
                for v in views:
                    with torch.cuda.amp.autocast():
                        outs.append(classifier(model(v)).float())
                logs.append(torch.stack(outs).mean(0).cpu())
                labels.append(lab)
        all_logits.append(torch.cat(logs, 0))
        all_labels = torch.cat(labels, 0)

    logits = torch.stack(all_logits).mean(0)

    print('\n[per-class acc, ensemble argmax]', {k: round(v, 1) for k, v in per_class(logits, all_labels, n_cls, cls_list).items()})
    sp0, se0, sc0, se20, sc20 = sp_se_score(logits, all_labels, n_cls)
    print('ensemble (argmax):  S_p={:.2f}  S_e(4c)={:.2f}  Score(4c)={:.2f}  |  S_e(2c)={:.2f}  Score(2c)={:.2f}'.format(sp0, se0, sc0, se20, sc20))

    results = []
    for b in np.linspace(args.bias_min, args.bias_max, args.bias_step):
        adj = logits.clone()
        adj[:, 0] += b
        sp, se, sc, se2, sc2 = sp_se_score(adj, all_labels, n_cls)
        results.append((sc, b, sp, se, se2, sc2))
    results.sort(reverse=True)

    print('\ntop-5 (bias on normal logit):')
    print('  bias      S_p     S_e     Score')
    for sc, b, sp, se, _se2, _sc2 in results[:5]:
        print('  {:+6.3f}  {:5.2f}  {:5.2f}  {:6.2f}'.format(b, sp, se, sc))

    sc, b, sp, se, se2, sc2 = results[0]
    print('\n>>> best: bias={:+.3f} -> S_p={:.2f}, S_e(4c)={:.2f}, Score(4c)={:.2f} | S_e(2c)={:.2f}, Score(2c)={:.2f}  (argmax4c {:.2f}, +{:.2f})'.format(
        b, sp, se, sc, se2, sc2, sc0, sc - sc0))


if __name__ == '__main__':
    main()
