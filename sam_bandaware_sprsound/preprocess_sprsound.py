"""
SPRSound 数据预处理脚本（患者级别划分，避免数据泄露）

从 JSON + WAV 文件生成 .npz 格式的训练数据

输出格式：
- X_train: [N, 128000] 音频数据
- y_train: [N] 标签（0-4）
- X_test: [M, 128000] 测试数据
- y_test: [M] 标签
"""

import os
import json
import numpy as np
import librosa
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

# 标签映射
SPRSOUND_LABELS = {
    'Normal': 0,
    'DAS': 1,
    'CAS': 2,
    'CAS & DAS': 3,
    'Poor Quality': 4
}

TARGET_SR = 16000
TARGET_DURATION = 8
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION  # 128000


def process_data(data_dir, split='train'):
    """处理 SPRSound 数据，返回患者ID、音频数据、标签"""
    json_dir = os.path.join(data_dir, f'{split}_classification_json')
    wav_dir = os.path.join(data_dir, f'{split}_classification_wav')

    patient_ids = []
    X, y = [], []

    for json_file in tqdm(os.listdir(json_dir), desc=f'处理 {split} 数据'):
        if not json_file.endswith('.json'):
            continue

        json_path = os.path.join(json_dir, json_file)
        wav_file = json_file.replace('.json', '.wav')
        wav_path = os.path.join(wav_dir, wav_file)

        if not os.path.exists(wav_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        label = data.get('record_annotation', 'Normal')
        if label not in SPRSOUND_LABELS:
            continue

        # 提取患者 ID（文件名第一个下划线前的部分）
        patient_id = json_file.split('_')[0]

        # 加载音频
        audio, sr = librosa.load(wav_path, sr=TARGET_SR)

        # 循环填充/截断
        if len(audio) < TARGET_SAMPLES:
            repeat_count = (TARGET_SAMPLES // len(audio)) + 1
            audio = np.tile(audio, repeat_count)[:TARGET_SAMPLES]
        else:
            audio = audio[:TARGET_SAMPLES]

        patient_ids.append(patient_id)
        X.append(audio)
        y.append(SPRSOUND_LABELS[label])

    return patient_ids, np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    data_dir = 'D:/tmp/class/SPRSound-main/SPRSound-main/Classification'

    print('=== 处理训练数据 ===')
    patient_ids, X, y = process_data(data_dir, 'train')
    print(f'总样本: {len(X)}, 标签分布: {Counter(y.tolist())}')
    print(f'唯一患者数: {len(set(patient_ids))}')

    # 按患者 ID 划分（避免数据泄露）
    print('\n=== 按患者 ID 划分数据集（80/20）===')
    unique_patients = list(set(patient_ids))
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )

    # 根据患者 ID 划分样本
    train_mask = np.array([pid in train_patients for pid in patient_ids])
    test_mask = np.array([pid in test_patients for pid in patient_ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f'训练集: {len(X_train)} 样本 ({len(train_patients)} 患者)')
    print(f'测试集: {len(X_test)} 样本 ({len(test_patients)} 患者)')
    print(f'训练集标签分布: {Counter(y_train.tolist())}')
    print(f'测试集标签分布: {Counter(y_test.tolist())}')

    # 验证没有患者重叠
    train_patient_set = set([patient_ids[i] for i in np.where(train_mask)[0]])
    test_patient_set = set([patient_ids[i] for i in np.where(test_mask)[0]])
    overlap = train_patient_set & test_patient_set
    print(f'\n患者重叠检查: {len(overlap)} 个重叠患者 (应为 0)')

    # 保存
    output_file = 'sprsound_data.npz'
    np.savez(output_file,
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print(f'\n数据已保存到: {output_file}')


if __name__ == '__main__':
    main()
