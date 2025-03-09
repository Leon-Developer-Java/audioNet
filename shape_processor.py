import torch
import torchaudio

class ShapeProcessor:
    """
    形状处理器
    
    仅调整音频数据的形状以符合模型输入要求[batch, 1, 48000]，不进行其他处理
    """
    def __init__(self, target_length=48000):
        self.target_length = target_length

    def to_mono(self, waveform):
        """转换为单声道音频"""
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def adjust_length(self, waveform):
        """调整音频长度到目标长度
        
        如果长度超过目标长度，则截取中间部分
        如果长度不足目标长度，则通过填充0来达到目标长度
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
    
    def apply_shape_processing(self, waveform):
        """应用形状处理
        
        仅调整音频形状以符合模型输入要求，不进行其他处理
        """
        # 转单声道
        waveform = self.to_mono(waveform)
        # 调整长度
        waveform = self.adjust_length(waveform)
        
        return waveform