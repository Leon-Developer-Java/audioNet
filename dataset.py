import os
import torchaudio
from torch.utils.data import Dataset
import torch

class AudioDataset(Dataset):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_dir = root_dir
        self.target_sample_rate = 16000  # 目标采样率设为16kHz
        
        # 收集所有音频文件
        self.audio_files = []
        self.labels = []
        self.label_names = []
        
        # 遍历音频目录
        for filename in os.listdir(self.audio_dir):
            if filename.endswith('.wav'):
                # 解析文件名获取标签信息
                # 0_dog_001_5.0s.wav -> class_idx=0, label_name='dog'
                parts = filename.split('_')
                class_idx = int(parts[0])
                label_name = parts[1]
                
                # 保存文件路径和标签信息
                self.audio_files.append(os.path.join(self.audio_dir, filename))
                self.labels.append(class_idx)
                self.label_names.append(label_name)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """获取指定索引的音频数据和标签
        
        Args:
            idx (int): 数据索引
            
        Returns:
            tuple: (音频数据, 类别标签)
                - 音频数据: torch.Tensor, 形状为[1, T]
                - 类别标签: int
        """
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(self.audio_files[idx])
        
        # 如果采样率不是目标采样率，进行重采样
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
            
        # 对音频数据进行标准化处理
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        waveform = (waveform - mean) / (std + 1e-6)  # 添加小值避免除零
        
        # 获取对应的标签
        label = self.labels[idx]
        
        return waveform, label
