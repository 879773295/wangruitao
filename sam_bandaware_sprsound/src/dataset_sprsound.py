"""
SPRSound 数据集加载器

SPRSound 数据集结构：
- train_classification_json/*.json - 标注文件
- train_classification_wav/*.wav - 音频文件
- valid_classification_json/*.json - 验证集标注
- valid_classification_wav/*.wav - 验证集音频

JSON 格式：
{
  "record_annotation": "Normal",  # 标签
  "event_annotation": [...]       # 事件标注（可选）
}

标签映射：
- Normal: 0
- DAS: 1
- CAS: 2
- CAS & DAS: 3
- Poor Quality: 4
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from transformers import ASTFeatureExtractor


# SPRSound 标签映射
SPRSOUND_LABELS = {
    'Normal': 0,
    'DAS': 1,
    'CAS': 2,
    'CAS & DAS': 3,
    'Poor Quality': 4
}

SPRSOUND_LABEL_NAMES = {v: k for k, v in SPRSOUND_LABELS.items()}


class SPRSoundDataset(Dataset):
    """
    SPRSound 数据集

    Args:
        data_dir: 数据根目录（包含 train/valid 子目录）
        split: 'train' 或 'valid'
        processor: ASTFeatureExtractor 实例
        target_sr: 目标采样率
        target_length: 目标长度（秒）
    """
    def __init__(self, data_dir, split='train', processor=None, target_sr=16000, target_length=8):
        self.data_dir = data_dir
        self.split = split
        self.target_sr = target_sr
        self.target_length = target_length
        self.target_samples = target_sr * target_length

        # 数据目录
        json_dir = os.path.join(data_dir, f'{split}_classification_json')
        wav_dir = os.path.join(data_dir, f'{split}_classification_wav')

        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"JSON 目录不存在: {json_dir}")
        if not os.path.exists(wav_dir):
            raise FileNotFoundError(f"WAV 目录不存在: {wav_dir}")

        # 读取所有样本
        self.samples = []
        for json_file in os.listdir(json_dir):
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
                print(f"警告：未知标签 {label}，跳过")
                continue

            self.samples.append({
                'wav_path': wav_path,
                'label': SPRSOUND_LABELS[label],
                'label_name': label,
                'json_path': json_path
            })

        print(f"加载 {split} 数据集: {len(self.samples)} 样本")

        # 统计标签分布
        from collections import Counter
        label_counts = Counter([s['label_name'] for s in self.samples])
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} ({count/len(self.samples)*100:.1f}%)")

        # 特征提取器
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载音频
        audio, sr = librosa.load(sample['wav_path'], sr=self.target_sr)

        # 循环填充/截断到目标长度
        if len(audio) < self.target_samples:
            # 循环填充
            repeat_count = (self.target_samples // len(audio)) + 1
            audio = np.tile(audio, repeat_count)[:self.target_samples]
        else:
            # 截断
            audio = audio[:self.target_samples]

        # 转换为 AST 输入格式
        if self.processor is not None:
            inputs = self.processor(
                audio,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs['input_values'].squeeze(0)  # [1, seq_len, freq]
        else:
            # 简单处理
            input_values = torch.tensor(audio).unsqueeze(0)  # [1, samples]

        return input_values, sample['label'], sample['label_name']
