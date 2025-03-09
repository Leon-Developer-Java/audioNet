import torch.nn as nn
import torch.nn.functional as F

class AemNetResidual(nn.Module):
    """音频环境声音分类网络（带残差连接）
    
    该网络采用两阶段特征提取架构：
    1. 低级特征提取（LLF）：使用1D卷积处理原始音频信号
    2. 高级特征提取（HLF）：使用2D卷积和残差连接处理特征图
    
    Args:
        num_classes (int): 分类类别数，默认51类
        wm (float): 网络宽度乘子，用于调整网络容量，默认0.5
    """
    def __init__(self, num_classes=51, wm=0.5):
        super(AemNetResidual, self).__init__()
        
        # LLF Block：低级特征提取模块
        # 输入音频信号: [batch, 1, 48000] -> 输出特征图: [batch, 128, 100]
        self.llf = nn.Sequential(
            # 第一个1D卷积层：时域特征提取
            # [batch, 1, 48000] -> [batch, 64, 24000]
            nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),  # 归一化处理
            nn.ReLU(),  # 非线性激活
            
            # 第二个1D卷积层：进一步压缩和特征提取
            # [batch, 64, 24000] -> [batch, 128, 12000]
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 最大池化层：降采样，提取最显著特征
            # [batch, 128, 12000] -> [batch, 128, 100]
            nn.MaxPool1d(kernel_size=10, stride=10)
        )

        # HLF Block：高级特征提取模块（带残差连接）
        # 输入特征图: [batch, 1, 128, 100] -> 输出类别预测: [batch, num_classes, 1, 1]
        self.hlf = nn.Sequential(
            # 初始特征映射层：将1通道特征图映射到多通道
            # [batch, 1, 128, 100] -> [batch, 32*wm, 128, 100]
            nn.Conv2d(1, int(32*wm), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(32*wm)),
            nn.ReLU(),
            # 空间降采样
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第一个残差块：特征通道扩展
            # [batch, 32*wm, 64, 50] -> [batch, 64*wm, 64, 50]
            self._make_residual_block(int(32*wm), int(64*wm)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个残差块：进一步特征提取
            # [batch, 64*wm, 32, 25] -> [batch, 128*wm, 32, 25]
            self._make_residual_block(int(64*wm), int(128*wm)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个残差块：高级特征提取
            # [batch, 128*wm, 16, 12] -> [batch, 256*wm, 16, 12]
            self._make_residual_block(int(128*wm), int(256*wm)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四个残差块：最终特征提取
            # [batch, 256*wm, 8, 6] -> [batch, 512*wm, 8, 6]
            self._make_residual_block(int(256*wm), int(512*wm)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 最终分类层：将特征图转换为类别预测
            # [batch, 512*wm, 4, 3] -> [batch, num_classes, 4, 3]
            nn.Conv2d(int(512*wm), num_classes, kernel_size=1),
            # 全局平均池化：压缩空间维度
            # [batch, num_classes, 4, 3] -> [batch, num_classes, 1, 1]
            nn.AdaptiveAvgPool2d(1)
        )
    def _make_residual_block(self, in_channels, out_channels):
        """创建残差块
        
        每个残差块包含两个3x3卷积层，并添加跳跃连接。如果输入输出通道数不同，
        会使用1x1卷积进行调整。
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            
        Returns:
            nn.Sequential: 残差块模块
        """
        return nn.Sequential(
            # 第一个卷积层：通道数调整
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # 第二个卷积层（带残差连接）
            ResidualAddWrapper(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )
    def forward(self, x):
        """前向传播函数
        
        Args:
            x (torch.Tensor): 输入音频特征，形状为[batch, 1, 48000]
            
        Returns:
            torch.Tensor: 类别预测概率，形状为[batch, num_classes]

        """
  
        # 低级特征提取
        # [batch, 1, 48000] -> [batch, 128, 100]
        # 重新初始化BatchNorm1d层
        if not hasattr(self, '_initialized_bn'):
            for m in self.llf.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.reset_running_stats()
                    m.reset_parameters()
            self._initialized_bn = True
            
        x = self.llf(x)
        
        # 维度转换：添加通道维度用于2D卷积
        # [batch, 128, 100] -> [batch, 1, 128, 100]
        x = x.unsqueeze(1)
        
        # 高级特征提取
        # [batch, 1, 128, 100] -> [batch, num_classes, 1, 1]
        x = self.hlf(x)
        
        # 压缩多余维度，得到最终预测
        # [batch, num_classes, 1, 1] -> [batch, num_classes]
        return x.squeeze()
class ResidualAddWrapper(nn.Module):
    """残差连接包装器
    
    将任意层序列包装成残差模块，通过跳跃连接实现残差学习。
    如果输入输出维度不匹配，使用1x1卷积进行调整。
    
    Args:
        *layers: 要包装的网络层序列
    """
    def __init__(self, *layers):
        super().__init__()
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播函数
        
        Args:
            x (torch.Tensor): 输入特征图
            
        Returns:
            torch.Tensor: 残差连接后的输出特征图
        """
        # 保存输入用于残差连接
        identity = x
        # 主路径前向传播
        out = self.block(x)
        
        # 确保维度匹配
        if out.shape != identity.shape:
            # 使用1x1卷积调整通道数，实现维度匹配
            identity = nn.Conv2d(identity.shape[1], out.shape[1], 
                               kernel_size=1).to(out.device)(identity)
        
        # 残差连接：将主路径输出与身份映射相加
        return out + identity