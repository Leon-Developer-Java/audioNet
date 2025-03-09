import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset

def create_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=4):
    """创建数据加载器
    
    Args:
        root_dir (str): 数据集根目录
        batch_size (int): 批次大小，默认32
        shuffle (bool): 是否打乱数据，默认True
        num_workers (int): 工作进程数，默认4
        
    Returns:
        DataLoader: PyTorch数据加载器对象
    """
    # 创建数据集实例
    dataset = AudioDataset(root_dir) # 开启数据增强 augment=True
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 使用锁页内存，加快数据传输
    )
    
    return dataloader

# # 使用示例
# if __name__ == '__main__':
#     # 创建数据加载器
#     train_loader = create_dataloader('e:')
    
#     # 遍历数据集
#     for batch_idx, (waveforms, labels) in enumerate(train_loader):
#         print(f'批次 {batch_idx}:')
#         print(f'\t波形形状: {waveforms.shape}')
#         print(f'\t标签形状: {labels.shape}')
        
#         # 只打印第一个批次
#         if batch_idx == 0:
#             break