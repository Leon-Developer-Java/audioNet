import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from dataloader import AudioDataset
from model_full import AemNet
from tqdm import tqdm
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
            
            current_loss = total_loss / (pbar.n + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

if __name__ == '__main__':
    # 创建数据集
    dataset = AudioDataset('audioSet/audio_data')
    batch_size = 32
    num_folds = 5
    
    # 准备分层K折交叉验证
    all_waveforms = []
    all_labels = []
    for i in range(len(dataset)):
        waveform, label = dataset[i]
        all_waveforms.append(waveform)
        all_labels.append(label)
    
    # 初始化分层K折交叉验证
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # 记录每个折的最佳验证准确率
    fold_best_accs = []
    
    # K折交叉验证训练
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_waveforms, all_labels)):
        print(f'\nFold {fold + 1}/{num_folds}')
        
        # 创建数据加载器
        train_loader = DataLoader(
            [dataset[i] for i in train_idx],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            [dataset[i] for i in val_idx],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 初始化模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AemNet(num_classes=51).to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-4)
        
        # 训练循环
        num_epochs = 200
        start_epoch = 0
        best_val_acc = 0
        
        # 创建保存模型的目录
        os.makedirs('checkpoints', exist_ok=True)
        
        # 检查是否存在检查点文件
        checkpoint_path = os.path.join('checkpoints', f'checkpoint_fold{fold+1}.pth')
        if os.path.exists(checkpoint_path):
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            print(f'Resuming training from epoch {start_epoch}')
        
        for epoch in range(start_epoch, num_epochs):
            # 训练阶段
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # 验证阶段
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # 打印训练信息
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join('checkpoints', f'best_model_fold{fold+1}.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            
            # 每个epoch结束后清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        fold_best_accs.append(best_val_acc)
        print(f'Fold {fold + 1} Best Validation Accuracy: {best_val_acc:.2f}%')
    
    # 打印所有折的平均准确率
    mean_acc = np.mean(fold_best_accs)
    std_acc = np.std(fold_best_accs)
    print(f'\nCross-validation results:')
    print(f'Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%')