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
        
        # 重采样到16kHz
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        # 标准化处理
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        waveform = (waveform - mean) / (std + 1e-6)  # 添加小值避免除零
        
        # 数据增强
        # 1. 随机噪声 (降低概率和噪声强度)
        if torch.rand(1) > 0.7:  # 将概率从0.5降低到0.3
            noise = torch.randn_like(waveform) * 0.003  # 将噪声强度从0.005降低到0.003
            waveform = waveform + noise
        
        # 2. 随机裁剪2秒片段
        target_length = int(2 * self.target_sample_rate)  # 2秒对应的采样点数
        if waveform.shape[1] > target_length:
            max_start = waveform.shape[1] - target_length
            start = torch.randint(0, max_start, (1,))
            waveform = waveform[:, start:start+target_length]
        else:
            # 如果音频长度不足2秒，则进行填充
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # 3. 随机增益变化 (缩小增益变化范围)
        if torch.rand(1) > 0.7:  # 将概率从0.5降低到0.3
            gain = 0.9 + 0.2 * torch.rand(1)  # 将增益范围从[0.8, 1.2]缩小到[0.9, 1.1]
            waveform = waveform * gain
        
        # 获取对应的标签
        label = self.labels[idx]
        
        return waveform, label
