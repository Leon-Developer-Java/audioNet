import torch
import torch.nn as nn
from torch.optim import Adam
from model_residual import AemNetResidual as AudioNet
from dataloader import create_dataloader
from progress import TrainProgressBar

def train(epochs=20, batch_size=64, learning_rate=0.0001):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = AudioNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 创建数据加载器
    train_loader = create_dataloader('e:', batch_size=batch_size)
    
    # ————————————————————初始化进度条管理器————————————————————
    progress_bar = TrainProgressBar(epochs, len(train_loader))
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            # 将数据移到设备上
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新batch进度条
            progress_bar.update_batch(batch_idx, loss.item())
        
        # ————————————————————计算平均损失并更新epoch进度条————————————————————
        avg_loss = total_loss / len(train_loader)
        progress_bar.update_epoch(epoch, avg_loss)
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'best_pth/model_epoch_{epoch+1}.pth')
    
    # 关闭进度条
    progress_bar.close()

if __name__ == '__main__':
    train(epochs=20,batch_size=8,learning_rate=0.0001)