import torch
from dataloader import create_dataloader
from tqdm import tqdm

def main():
    # 创建数据加载器，使用较小的batch_size
    train_loader = create_dataloader('e:/audioNet/audioSet/audio_data', batch_size=4)
    
    # 遍历数据集
    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        print(f'\n批次 {batch_idx}:')
        print(f'音频数据shape: {waveforms.shape}  # [batch_size, channels, time_steps]')
        print(f'标签: {labels}')
        
        # 只打印前3个批次
        if batch_idx >= 2:
            break

if __name__ == '__main__':
    main()