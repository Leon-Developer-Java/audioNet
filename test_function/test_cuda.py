import torch
import sys
import os

# 添加父目录到系统路径以导入model_residual
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_residual import AemNetResidual

def test_cuda():
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA是否可用: {cuda_available}")
    
    if cuda_available:
        # 获取CUDA设备信息
        device_count = torch.cuda.device_count()
        print(f"可用的CUDA设备数量: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"CUDA设备 {i}: {device_name}")
            
        # 获取当前CUDA设备信息
        current_device = torch.cuda.current_device()
        print(f"当前CUDA设备索引: {current_device}")
        
        # 获取显存信息
        print(f"当前设备显存分配: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB")
        print(f"当前设备显存缓存: {torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB")
    
    # 测试模型在不同设备上的运行
    print("\n测试模型在不同设备上的运行:")
    
    # 创建一个小的随机输入
    x = torch.randn(4, 1, 4000)
    print(f"输入张量设备: {x.device}")
    
    # 在CPU上创建模型
    model_cpu = AemNetResidual()
    print(f"CPU模型设备: {next(model_cpu.parameters()).device}")
    
    if cuda_available:
        # 将模型移到GPU
        model_gpu = AemNetResidual().cuda()
        print(f"GPU模型设备: {next(model_gpu.parameters()).device}")
        
        # 将输入移到GPU
        x_gpu = x.cuda()
        print(f"GPU输入张量设备: {x_gpu.device}")
        
        # 在GPU上进行前向传播
        with torch.no_grad():
            output_gpu = model_gpu(x_gpu)
        print(f"GPU输出张量设备: {output_gpu.device}")

if __name__ == '__main__':
    test_cuda()