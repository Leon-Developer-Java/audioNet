import torch
import sys
import os

# 添加父目录到系统路径以导入model_residual
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_residual import AemNetResidual

def test_model_layers():
    # 创建模型实例
    model = AemNetResidual(num_classes=51, wm=0.5)
    model.eval()
    
    # 创建一个随机输入张量 [batch_size=4, channels=1, length=4000]
    x = torch.randn(4, 1, 4000)
    print("\n输入张量形状:", x.shape)
    
    # 测试LLF Block的输出
    llf_out = model.llf(x)
    print("\nLLF Block输出形状:", llf_out.shape)
    
    # 维度转换
    x_unsqueezed = llf_out.unsqueeze(1)
    print("\n维度转换后形状:", x_unsqueezed.shape)
    
    # 获取HLF Block中的各层
    hlf_layers = list(model.hlf.children())
    
    # 逐层测试HLF Block
    current_output = x_unsqueezed
    print("\nHLF Block各层输出形状:")
    for i, layer in enumerate(hlf_layers):
        current_output = layer(current_output)
        print(f"第{i+1}层: {current_output.shape}")
    
    # 最终输出
    final_output = current_output.squeeze()
    print("\n最终输出形状:", final_output.shape)

if __name__ == '__main__':
    test_model_layers()