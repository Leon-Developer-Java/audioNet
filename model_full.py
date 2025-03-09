import torch
import torch.nn as nn
import torch.nn.functional as F

class AemNet(nn.Module):
    def __init__(self,  num_classes=50,wm=1.0,):
        super(AemNet, self).__init__()
        
        # LLF Block (低级特征提取)
        self.llf = nn.Sequential(
            # Conv1: 1 * 4000 -> 64 * 2000 (修正后的实际输出)
            nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Conv2: 64 * 2000 -> 128 * 1000
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # MaxPool1: 128 * 996 -> 128 * 100
            nn.MaxPool1d(kernel_size=10, stride=10)
        )

        # HLF Block (高级特征提取)
        self.hlf = nn.Sequential(
            # 维度转换 [batch, 128, 100] -> [batch, 1, 128, 100]
            nn.Conv2d(1, int(32*wm), kernel_size=3, padding=1),  # Conv3
            nn.BatchNorm2d(int(32*wm)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool2
            
            # Conv4: 32W*50 -> 64W*50
            nn.Conv2d(int(32*wm), int(64*wm), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(64*wm)),
            nn.ReLU(),
            
            # Conv5: 64W*50 -> 64W*50
            nn.Conv2d(int(64*wm), int(64*wm), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(64*wm)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool3
            
            # Conv6: 64W*25 -> 128W*25
            nn.Conv2d(int(64*wm), int(128*wm), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(128*wm)),
            nn.ReLU(),
            
            # Conv7: 128W*25 -> 128W*25
            nn.Conv2d(int(128*wm), int(128*wm), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(128*wm)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool4
            
            # Conv8: 128W*12 -> 256W*12
            nn.Conv2d(int(128*wm), int(256*wm), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256*wm)),
            nn.ReLU(),
            
            # Conv9: 256W*12 -> 256W*12
            nn.Conv2d(int(256*wm), int(256*wm), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256*wm)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool5
            
            # Conv10: 256W*6 -> 512W*6
            nn.Conv2d(int(256*wm), int(512*wm), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(512*wm)),
            nn.ReLU(),
            
            # Conv11: 512W*6 -> 512W*6
            nn.Conv2d(int(512*wm), int(512*wm), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(512*wm)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool6
            
            # Conv12: 512W*3 -> 50
            nn.Conv2d(int(512*wm), num_classes, kernel_size=1),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # 输入形状: [batch, 1, 4000]
        x = self.llf(x)
        
        # 维度转换 [batch, 128, 100] -> [batch, 1, 128, 100]
        x = x.unsqueeze(1)
        
        # HLF处理
        x = self.hlf(x)
        
        # 输出形状: [batch, 50]
        return x.squeeze()