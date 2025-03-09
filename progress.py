from tqdm import tqdm

class TrainProgressBar:
    """训练进度条管理类
    
    用于管理训练过程中的进度显示，包括epoch和batch两个层级的进度条。
    """
    
    def __init__(self, total_epochs, total_batches):
        """初始化进度条管理器
        
        Args:
            total_epochs (int): 总的训练轮数
            total_batches (int): 每轮训练中的总批次数
        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        
        # 初始化epoch级进度条
        self.epoch_pbar = tqdm(
            total=total_epochs,
            desc='Training Progress',
            position=0,
            leave=True
        )
        
        # 初始化batch级进度条
        self.batch_pbar = tqdm(
            total=total_batches,
            desc='Current Epoch',
            position=1,
            leave=True
        )
    
    def update_epoch(self, epoch, avg_loss):
        """更新epoch进度条
        
        Args:
            epoch (int): 当前的训练轮数
            avg_loss (float): 当前epoch的平均损失值
        """
        # 更新epoch进度条
        self.epoch_pbar.set_description(f'Training Progress (Avg Loss: {avg_loss:.4f})')
        self.epoch_pbar.update(1)
        
        # 重置batch进度条
        self.batch_pbar.reset()
    
    def update_batch(self, batch_idx, loss):
        """更新batch进度条
        
        Args:
            batch_idx (int): 当前批次索引
            loss (float): 当前批次的损失值
        """
        # 更新batch进度条
        self.batch_pbar.set_description(f'Current Epoch (Loss: {loss:.4f})')
        self.batch_pbar.update(1)
    
    def close(self):
        """关闭所有进度条"""
        self.epoch_pbar.close()
        self.batch_pbar.close()