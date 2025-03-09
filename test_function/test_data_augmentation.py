import sys
sys.path.append('e:/audioNet')

import os
import torch
from dataset import AudioDataset

def visualize_augmentation():
    # 创建带数据增强的数据集
    augmented_dataset = AudioDataset(root_dir='e:/audioNet')
    
    # 创建原始数据集
    raw_dataset = AudioDataset(root_dir='e:/audioNet')

    # 随机选择一个样本
    idx = torch.randint(0, len(raw_dataset), (1,)).item()
    
    # 获取原始和增强后的数据
    raw_waveform, label = raw_dataset[idx]
    aug_waveform, _ = augmented_dataset[idx]

    # 打印形状信息
    print(f'原始音频形状: {raw_waveform.shape} (类别: {raw_dataset.label_names[label]})')
    print(f'增强后音频形状: {aug_waveform.shape}')

if __name__ == "__main__":
    visualize_augmentation()