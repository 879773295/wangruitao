"""
SPRSound 事件级 7 分类预处理脚本（原始 SAM 版）

从 BioCAS2022 的 JSON + WAV 提取每个事件（event_annotation），
重采样到 16kHz，循环填充到 8s，保存为 .npz。

官方划分：train2022（1949 条）为训练集，
          test2022（intra 379 + inter 355 = 734 条）为测试集。

7 类标签映射：
    Normal=0, Rhonchi=1, Wheeze=2, Stridor=3,
    Coarse Crackle=4, Fine Crackle=5, Wheeze+Crackle=6

输出：
    X_train: [N, 128000] float32
    y_train: [N] int64
    device_train: [N] int64（全部 -1，SPRSound 无设备信息）
    X_test / y_test / device_test 同理
"""

import os
import json
import numpy as np
import librosa
from tqdm import tqdm
from collections import Counter

DATA_DIR = "D:/tmp/class/SPRSound-main/SPRSound-main/BioCAS2022"
OUTPUT_FILENAME = "sprsound_event_7class.npz"

TARGET_SR = 16000
TARGET_DURATION = 8
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION  # 128000

# 7 类事件标签
EVENT_LABELS = {
    'Normal': 0,
    'Rhonchi': 1,
    'Wheeze': 2,
    'Stridor': 3,
    'Coarse Crackle': 4,
    'Fine Crackle': 5,
    'Wheeze+Crackle': 6,
}

MIN_EVENT_MS = 100  # 忽略过短的事件（数据中最小为 126ms）


def cyclic_padding(wav, target_len):
    curr_len = len(wav)
    if curr_len >= target_len:
        return wav[:target_len]
    repeat_count = (target_len // curr_len) + 1
    padded = np.tile(wav, repeat_count)
    return padded[:target_len]


def process_json(json_path, wav_dir):
    """处理一个 JSON，返回该记录所有事件的 (音频, 标签) 列表"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    events = data.get('event_annotation', [])
    if not events:
        return []

    wav_file = os.path.basename(json_path).replace('.json', '.wav')
    wav_path = os.path.join(wav_dir, wav_file)
    if not os.path.exists(wav_path):
        return []

    audio, _ = librosa.load(wav_path, sr=TARGET_SR)

    samples = []
    for ev in events:
        start_ms = int(ev['start'])
        end_ms = int(ev['end'])
        etype = ev['type']
        if etype not in EVENT_LABELS:
            continue
        if end_ms - start_ms < MIN_EVENT_MS:
            continue

        start = int(start_ms / 1000.0 * TARGET_SR)
        end = int(end_ms / 1000.0 * TARGET_SR)
        chunk = audio[start:end]
        if len(chunk) < 1:
            continue

        processed = cyclic_padding(chunk, TARGET_SAMPLES)
        samples.append((processed, EVENT_LABELS[etype]))

    return samples


def process_split(json_dirs, wav_dir, desc):
    """处理一个划分（train 或 test），收集所有事件样本"""
    X, y = [], []
    json_files = []
    for jd in json_dirs:
        if os.path.isdir(jd):
            json_files += [os.path.join(jd, f) for f in os.listdir(jd) if f.endswith('.json')]

    for jp in tqdm(json_files, desc=desc):
        for audio, label in process_json(jp, wav_dir):
            X.append(audio)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    print("=== SPRSound 事件级 7 分类预处理 ===")

    # 训练集
    train_json = os.path.join(DATA_DIR, 'train2022_json')
    train_wav = os.path.join(DATA_DIR, 'train2022_wav')
    X_train, y_train = process_split([train_json], train_wav, '训练集')

    # 测试集（intra + inter）
    test_json_intra = os.path.join(DATA_DIR, 'test2022_json', 'intra_test_json')
    test_json_inter = os.path.join(DATA_DIR, 'test2022_json', 'inter_test_json')
    test_wav = os.path.join(DATA_DIR, 'test2022_wav')
    X_test, y_test = process_split([test_json_intra, test_json_inter], test_wav, '测试集')

    device_train = np.full(len(y_train), -1, dtype=np.int64)
    device_test = np.full(len(y_test), -1, dtype=np.int64)

    print(f"\n训练集: {X_train.shape}  标签分布: {dict(sorted(Counter(y_train.tolist()).items()))}")
    print(f"测试集: {X_test.shape}  标签分布: {dict(sorted(Counter(y_test.tolist()).items()))}")

    np.savez(OUTPUT_FILENAME,
             X_train=X_train, y_train=y_train, device_train=device_train,
             X_test=X_test, y_test=y_test, device_test=device_test)

    print(f"\n已保存: {OUTPUT_FILENAME}")


if __name__ == '__main__':
    main()
