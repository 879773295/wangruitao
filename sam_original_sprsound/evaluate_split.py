"""
在已训练的 best_model.pth 上，分别评估 inter-only（无泄露）和 intra-only（有泄露）

inter-test：41 患者，0% 与训练集重叠 → 真实泛化
intra-test：162 患者，100% 与训练集重叠 → 患者内表现 / 泄露上界

输出：每个子集的 S_p / S_e / Score / Acc / 各类别准确率
"""

import os
import json
import numpy as np
import torch
import librosa
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from transformers import ASTFeatureExtractor

from train_event import CustomAST, CLASS_NAMES, per_class_accuracy

DATA_DIR = "D:/tmp/class/SPRSound-main/SPRSound-main/BioCAS2022"
TARGET_SR = 16000
TARGET_SAMPLES = TARGET_SR * 8  # 128000
MIN_EVENT_MS = 100

EVENT_LABELS = {
    'Normal': 0, 'Rhonchi': 1, 'Wheeze': 2, 'Stridor': 3,
    'Coarse Crackle': 4, 'Fine Crackle': 5, 'Wheeze+Crackle': 6,
}


def cyclic_padding(wav, target_len):
    curr_len = len(wav)
    if curr_len >= target_len:
        return wav[:target_len]
    return np.tile(wav, (target_len // curr_len) + 1)[:target_len]


def load_events(json_dirs, wav_dir):
    """提取事件样本，返回 (X, y, patient_ids)"""
    X, y, pids = [], [], []
    json_files = []
    for jd in json_dirs:
        json_files += [os.path.join(jd, f) for f in os.listdir(jd) if f.endswith('.json')]

    for jp in tqdm(json_files, desc='提取事件', leave=False):
        with open(jp, 'r') as f:
            data = json.load(f)
        events = data.get('event_annotation', [])
        if not events:
            continue
        wav_path = os.path.join(wav_dir, os.path.basename(jp).replace('.json', '.wav'))
        if not os.path.exists(wav_path):
            continue
        audio, _ = librosa.load(wav_path, sr=TARGET_SR)
        pid = os.path.basename(jp).split('_')[0]
        for ev in events:
            etype = ev['type']
            if etype not in EVENT_LABELS:
                continue
            if int(ev['end']) - int(ev['start']) < MIN_EVENT_MS:
                continue
            s = int(int(ev['start']) / 1000.0 * TARGET_SR)
            e = int(int(ev['end']) / 1000.0 * TARGET_SR)
            chunk = audio[s:e]
            if len(chunk) < 1:
                continue
            X.append(cyclic_padding(chunk, TARGET_SAMPLES))
            y.append(EVENT_LABELS[etype])
            pids.append(pid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), pids


@torch.no_grad()
def evaluate(model, processor, X, y, batch_size=8):
    device = next(model.parameters()).device
    preds = []
    for i in tqdm(range(0, len(X), batch_size), desc='评估', leave=False):
        batch = X[i:i+batch_size]
        inp = torch.stack([processor(w, sampling_rate=16000, return_tensors='pt').input_values.squeeze(0) for w in batch]).to(device)
        logits = model(inp)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return confusion_matrix(y, np.array(preds))


def report(name, cm):
    n = cm.shape[0]
    sp = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
    se = cm[1:, 1:].sum() / cm[1:, :].sum() if cm[1:, :].sum() > 0 else 0
    acc = np.diag(cm).sum() / cm.sum()
    pca = per_class_accuracy(cm)
    print(f'\n=== {name} ===  样本数: {int(cm.sum())}')
    print(f'  S_p: {sp*100:.2f}  S_e: {se*100:.2f}  Score: {(sp+se)/2*100:.2f}  Acc: {acc*100:.2f}')
    pca_str = '  '.join(f'{CLASS_NAMES[i]}={pca[i]*100:.1f}' for i in range(n))
    print(f'  各类: {pca_str}')


def main():
    device = 'cuda'
    print('加载模型...')
    model = CustomAST(num_classes=7).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()

    processor = ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')

    test_wav = os.path.join(DATA_DIR, 'test2022_wav')
    inter_dir = os.path.join(DATA_DIR, 'test2022_json', 'inter_test_json')
    intra_dir = os.path.join(DATA_DIR, 'test2022_json', 'intra_test_json')

    X_inter, y_inter, _ = load_events([inter_dir], test_wav)
    X_intra, y_intra, _ = load_events([intra_dir], test_wav)

    print(f'inter 事件数: {len(y_inter)}, intra 事件数: {len(y_intra)}')

    cm_inter = evaluate(model, processor, X_inter, y_inter)
    report('inter-only（无泄露，真实泛化）', cm_inter)

    cm_intra = evaluate(model, processor, X_intra, y_intra)
    report('intra-only（有泄露，患者内）', cm_intra)

    cm_all = cm_inter + cm_intra
    report('合并（inter+intra）', cm_all)


if __name__ == '__main__':
    main()
