import torch
import torchaudio

class BasicProcessor:
    """基础音频处理器
    
    仅执行必要的预处理操作，不进行数据增强，确保输出与增强数据具有相同的shape
    """
    def __init__(self, sample_rate=16000, target_length=48000):
        self.sample_rate = sample_rate
        self.target_length = target_length  # 目标长度（3秒@16kHz）
        self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)
    
    def normalize(self, waveform):
        """音频幅度标准化处理"""
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.std() + 1e-8)
        return waveform
    
    def to_mono(self, waveform):
        """转换为单声道音频"""
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def center_crop(self, waveform):
        """中心裁剪音频到目标长度
        
        与随机裁剪不同，这里使用中心裁剪以保持一致性
        """
        if waveform.size(1) > self.target_length:
            # 计算中心点
            start = (waveform.size(1) - self.target_length) // 2
            # 裁剪音频：保留所有通道，截取目标长度的片段
            return waveform[:, start:start+self.target_length]
        elif waveform.size(1) < self.target_length:
            # 如果音频长度不足，通过填充0来达到目标长度
            padding = self.target_length - waveform.size(1)
            padding_left = padding // 2
            padding_right = padding - padding_left
            return torch.nn.functional.pad(waveform, (padding_left, padding_right))
        # 长度刚好，直接返回
        return waveform
    
    def apply_basic_processing(self, waveform):
        """应用基础处理流程
        
        执行必要的预处理，但不进行数据增强
        """
        # 重采样
        waveform = self.resample(waveform)
        # # 转单声道
        # waveform = self.to_mono(waveform)
        # # 标准化
        # waveform = self.normalize(waveform)
        # # 中心裁剪到固定长度
        # waveform = self.center_crop(waveform)
        
        return waveform