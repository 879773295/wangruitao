"""
混淆矩阵绘图脚本（论文级，仿 SPRSound 论文 Fig.13 风格）

三种输入方式（任选其一）：

1) 手动输入数字（最简单）：
   编辑一个 JSON 文件（模板见 matrix_example.json），然后：
       python plot_confusion.py --json my_matrix.json

2) SPRSound 事件级自动计算（用已训练的 checkpoint 跑测试集）：
       python plot_confusion.py --sprsound --split inter          # 患者独立测试
       python plot_confusion.py --sprsound --split intra
       python plot_confusion.py --sprsound --split all            # 合并

3) 预测结果文件（y_true + y_pred，适合 ICBHI 或任何任务）：
   npz 文件里需包含两个数组：y_true（真实标签）、y_pred（预测标签），然后：
       python plot_confusion.py --preds preds.npz --labels Normal Crackle Wheeze Both

可选参数：
    --title "..."        图标题（默认按输入方式自动生成）
    --out fig.png         输出文件名（默认 conf_<来源>.png，也支持 .pdf）
    --style fig13|blues   配色：fig13=对角线蓝/红+白底（默认，仿原论文）；blues=经典蓝色渐变
    --normal-index 0      "Normal" 类的下标（决定对角线用蓝色，其余对角线用红色）
    --figsize 6           图片宽度（英寸），高度按类别数自动算
    --binary              折叠成 2×2（Normal vs Adventitious，对应论文 Task 1-1 口径）：
                          TN=正常→正常, FP=正常→异常, FN=异常→正常, TP=异常→异常
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体（Windows），避免标题/标签出现方块
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 输入方式 1：JSON 手动输入
# ---------------------------------------------------------------------------
def load_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = np.array(data['counts'], dtype=int)
    labels = data.get('labels', [str(i) for i in range(counts.shape[0])])
    title = data.get('title', os.path.splitext(os.path.basename(path))[0])
    return counts, labels, title


# ---------------------------------------------------------------------------
# 输入方式 2：SPRSound 事件级自动计算
# ---------------------------------------------------------------------------
def load_from_sprsound(ckpt_path, split):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam_original_sprsound'))
    import torch
    from transformers import ASTFeatureExtractor
    from train_event import CustomAST, CLASS_NAMES
    from evaluate_split import load_events, evaluate, DATA_DIR

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'加载模型: {ckpt_path}')
    model = CustomAST(num_classes=7).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    processor = ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')

    wav_dir = os.path.join(DATA_DIR, 'test2022_wav')
    json_dirs = {
        'inter': [os.path.join(DATA_DIR, 'test2022_json', 'inter_test_json')],
        'intra': [os.path.join(DATA_DIR, 'test2022_json', 'intra_test_json')],
    }
    if split == 'all':
        dirs = json_dirs['intra'] + json_dirs['inter']
    else:
        dirs = json_dirs[split]

    X, y, _ = load_events(dirs, wav_dir)
    print(f'{split} 事件数: {len(y)}，开始推理...')
    cm = evaluate(model, processor, X, y)
    title = f'SPRSound event-level 7-class ({split})'
    return np.array(cm, dtype=int), CLASS_NAMES, title


# ---------------------------------------------------------------------------
# 输入方式 3：preds.npz（y_true + y_pred）
# ---------------------------------------------------------------------------
def load_from_preds(path, labels):
    data = np.load(path)
    y_true = np.asarray(data['y_true'])
    y_pred = np.asarray(data['y_pred'])
    n = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    if labels is None:
        labels = [str(i) for i in range(n)]
    title = os.path.splitext(os.path.basename(path))[0]
    return cm, labels, title


# ---------------------------------------------------------------------------
# 折叠成 2×2：Normal（下标 normal_index）vs Adventitious（其余全部合并）
# 行=真实标签，列=预测标签：
#   [[TN, FP],
#    [FN, TP]]
# ---------------------------------------------------------------------------
def to_binary(counts, labels, normal_index=0):
    i = normal_index
    tn = int(counts[i, i])
    fp = int(counts[i, :].sum() - tn)
    fn = int(counts[:, i].sum() - tn)
    tp = int(counts.sum() - tn - fp - fn)
    return np.array([[tn, fp], [fn, tp]], dtype=int), ['Normal', 'Adventitious']


# ---------------------------------------------------------------------------
# 绘图（仿 Fig.13：对角线 Normal=蓝、其余=红，白底，格子里写数量 + 对角线百分比）
# ---------------------------------------------------------------------------
def plot_confusion(counts, labels, title, out, style='fig13',
                   normal_index=0, figsize=6):
    """仿 SPRSound 论文 Fig.13 版式：
    - Predicted Label + 类别名在顶部横排，True Label 与类别名在左侧竖排
    - 对角线格子：数量在上、(xx%) 在下两行；Normal 类蓝色、异常类红色
    - 底部图例：蓝=正确识别正常音，红=正确识别异常音
    """
    n = len(labels)
    row_sum = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts * 100.0, np.where(row_sum == 0, 1, row_sum))

    # 类别多时自动加宽，避免标签重叠
    fig = plt.figure(figsize=(max(figsize, 0.85 * n + 2), max(figsize, 0.85 * n + 2)))
    ax = fig.add_subplot(111)

    if style == 'blues':
        ax.imshow(pct, cmap='Blues')
        text_color = lambda r, c: 'white' if pct[r, c] > 50 else 'black'
    else:
        from matplotlib.colors import ListedColormap
        ax.imshow(np.zeros((n, n)), cmap=ListedColormap(['white']))
        for i in range(n):
            color = '#3b5bdb' if i == normal_index else '#e03131'
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, edgecolor='#adb5bd', zorder=1))
        text_color = lambda r, c: 'white' if r == c else 'black'

    # 格子文字：对角线数量在上、(xx%) 换行在下；其余只有数量
    for i in range(n):
        for j in range(n):
            if i == j:
                tag = f'{counts[i, j]}\n({pct[i, j]:.0f}%)'
            else:
                tag = str(counts[i, j])
            ax.text(j, i, tag, ha='center', va='center', fontsize=9,
                    color=text_color(i, j), linespacing=1.2, zorder=2)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # 类别名：顶部横排（在空格和 + 处换行防重叠），左侧竖排 —— 与论文一致
    def wrap(label):
        for sep in [' ', '+']:
            if sep in label:
                return label.replace(sep, '\n', 1)
        return label

    top_labels = [wrap(l) for l in labels]
    ax.set_xticklabels(top_labels, fontsize=8)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_yticklabels(top_labels, rotation=90, fontsize=8)
    ax.set_xlabel('Predicted Label', fontsize=10, labelpad=6)
    ax.set_ylabel('True Label', fontsize=10, labelpad=10)
    if title:
        ax.set_title(title, fontsize=11, pad=34 if title else 10)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='#adb5bd', linewidth=0.8)
    ax.tick_params(which='minor', length=0)

    # 图例（fig13 风格：蓝=正确正常音，红=正确异常音）
    if style == 'fig13':
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor='#3b5bdb', edgecolor='#adb5bd',
                  label='Correctly Recognized Normal Sounds'),
            Patch(facecolor='#e03131', edgecolor='#adb5bd',
                  label='Correctly Recognized Adventitious Sounds'),
        ]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -0.06),
                  ncol=1, fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {out}')


def main():
    ap = argparse.ArgumentParser(description='画论文级混淆矩阵')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--json', help='手动输入的 JSON 文件（模板 matrix_example.json）')
    src.add_argument('--sprsound', action='store_true', help='SPRSound 事件级：自动跑测试集')
    src.add_argument('--preds', help='npz 文件，含 y_true / y_pred 两个数组')

    ap.add_argument('--split', choices=['inter', 'intra', 'all'], default='inter',
                    help='SPRSound 测试子集（默认 inter）')
    ap.add_argument('--ckpt', default='sam_original_sprsound/checkpoints/best_model.pth',
                    help='SPRSound 模型权重路径')
    ap.add_argument('--labels', nargs='*', help='类别名（preds 模式使用）')
    ap.add_argument('--title', help='图标题')
    ap.add_argument('--out', help='输出文件（默认 conf_<来源>.png / .pdf）')
    ap.add_argument('--style', choices=['fig13', 'blues'], default='fig13')
    ap.add_argument('--normal-index', type=int, default=0, help='Normal 类下标')
    ap.add_argument('--figsize', type=float, default=6, help='图片宽度（英寸）')
    ap.add_argument('--binary', action='store_true',
                    help='折叠成 2×2（Normal vs Adventitious，仿论文 Task 1-1）')
    args = ap.parse_args()

    if args.json:
        counts, labels, title = load_from_json(args.json)
        default_out = os.path.splitext(args.json)[0] + '.png'
    elif args.sprsound:
        counts, labels, title = load_from_sprsound(args.ckpt, args.split)
        default_out = f'conf_sprsound_{args.split}.png'
    else:
        counts, labels, title = load_from_preds(args.preds, args.labels)
        default_out = 'conf_' + os.path.splitext(os.path.basename(args.preds))[0] + '.png'

    if args.binary:
        counts, labels = to_binary(counts, labels, args.normal_index)
        if not args.title:
            title = title + ' (binary)'
        if args.out is None and not default_out.endswith('_binary.png'):
            base = os.path.splitext(default_out)[0]
            default_out = base + '_binary.png'

    if args.title:
        title = args.title
    out = args.out or default_out

    print('混淆矩阵（行=真实标签）:')
    print('        ' + ' '.join(f'{l:>8.8}' for l in labels))
    for l, row in zip(labels, counts):
        print(f'{l:>8.8} ' + ' '.join(f'{v:>8d}' for v in row))

    plot_confusion(counts, labels, title, out, style=args.style,
                   normal_index=args.normal_index, figsize=args.figsize)


if __name__ == '__main__':
    main()
