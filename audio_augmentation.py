import torch
import torchaudio
import random
import numpy as np

class AugmentationProcessor:
    # 音频增强处理器初始化
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)

        # 音频幅度标准化处理
    def normalize(self, waveform):
        # 幅度标准化
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.std() + 1e-8)
        return waveform

        # 添加随机高斯噪声
    def add_random_noise(self, waveform):
        # 降低噪声强度，从0.005降低到0.003，使增强更温和
        noise = torch.randn_like(waveform) * 0.003
        return waveform + noise

        # 随机裁剪音频片段
    def random_crop(self, waveform, target_length=48000):
        # 检查音频长度是否超过目标长度（3秒@16kHz）
        if waveform.size(1) > target_length:
            # 在有效范围内随机生成起始点：[0, 总长度-目标长度]
            start = random.randint(0, waveform.size(1) - target_length)
            # 裁剪音频：保留所有通道，截取目标长度的片段
            return waveform[:, start:start+target_length]
        # 音频长度不足时直接返回原波形
        return waveform

        # 随机增益调节
    def random_gain(self, waveform):
        # 缩小增益范围，使增强更温和
        gain = random.uniform(0.9, 1.1)
        return waveform * gain

        # 转换为单声道音频
    def to_mono(self, waveform):
        # 多声道转单声道
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


        # 混合增强：线性组合两个音频样本
    def mixup_augmentation(self, waveform1, waveform2, alpha=0.4):
        """混合增强技术"""
        lam = np.random.beta(alpha, alpha)
        mixed_waveform = lam * waveform1 + (1 - lam) * waveform2
        return mixed_waveform

    def apply_basic(self, waveform):
        """应用基础处理流程
        
        执行必要的预处理，但不进行数据增强
        """
        # 重采样
        waveform = self.resample(waveform)
        
        return waveform

        # 应用完整增强流水线
    def apply_augmentations(self, waveform):
        # 基础预处理
        waveform = self.resample(waveform)

        # 随机应用增强技术，但降低应用概率，使增强更温和
        if random.random() > 0.7:  # 降低噪声添加概率
            waveform = self.add_random_noise(waveform)
        
        
        if random.random() > 0.7:  # 降低增益调整概率
            waveform = self.random_gain(waveform)

        return waveform
