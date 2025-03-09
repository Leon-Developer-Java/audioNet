import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from dataloader import AudioDataset
import pandas as pd
from model_residual import AemNetResidual as AemNet
from tqdm import tqdm
import multiprocessing
from visualization import plot_training_history, plot_learning_rate
# 移除不再需要的StratifiedKFold导入
import numpy as np

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for waveforms, labels in pbar:
        # print(f'音频数据shape: {waveforms.shape}  # [batch_size, channels, time_steps]')
        waveforms, labels = waveforms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条信息
        current_loss = total_loss / (pbar.n + 1)
        current_acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

# 验证函数
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条信息
            current_loss = total_loss / (pbar.n + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

if __name__ == '__main__':
    # Windows系统下需要添加多进程支持
    multiprocessing.freeze_support()
    
    # 创建数据集
    dataset = AudioDataset('audioSet/audio_data')
    # 使用简单的8:2随机划分数据集
    batch_size = 32  # 批次大小
    
    # 计算训练集和验证集的大小
    dataset_size = len(dataset)
  
    train_size = int(dataset_size * 0.8)  # 80%用于训练
    val_size = dataset_size - train_size  # 20%用于验证
    

    # 首先划分训练集和验证集
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子确保可重复性
    )

    

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                            pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                          pin_memory=True, persistent_workers=False)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AemNet(num_classes=51, wm=0.5).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-4)

    # 设置学习率调度器
    warmup_epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=200 - warmup_epochs,
        T_mult=1,
        eta_min=1e-6
    )
    # 训练循环
    num_epochs = 100
    best_val_acc = 0
    
    # 用于记录训练历史的列表
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    learning_rates = []
    
    # 创建保存图表的目录
    os.makedirs('plots', exist_ok=True)
    
    # 创建保存模型的目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 检查是否有检查点可以恢复训练
    start_epoch = 0
    checkpoint_path = os.path.join('checkpoints', 'training_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accs = checkpoint['train_accs']
        val_accs = checkpoint['val_accs']
        learning_rates = checkpoint['learning_rates']
        print(f'恢复训练从epoch {start_epoch+1}')
    # 训练循环中的训练阶段
    for epoch in tqdm(range(start_epoch, num_epochs), desc='Training Progress', position=0):
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step()
        else:
            # 预热阶段的学习率调整
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4 * warmup_factor
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录训练历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印训练信息
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join('checkpoints', f'best_model_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%')
        
        # 保存训练检查点，用于恢复训练
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'learning_rates': learning_rates
        }
        torch.save(checkpoint, os.path.join('checkpoints', 'training_checkpoint.pth'))
        
        print('-' * 60)
        
        # 每个epoch结束后清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # 训练完成后进行可视化
    from visualization import plot_training_history, plot_learning_rate
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 绘制学习率变化曲线
    plot_learning_rate(learning_rates)
    
    print('Training finished!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')