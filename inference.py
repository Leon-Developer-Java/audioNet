import os
import torch
import torch.nn.functional as F
import json
from dataset import AudioDataset
from model_residual import AemNetResidual as AemNet

def inference(model_path, test_data_dir):
    # 加载标签映射
    with open('label_map.json', 'r') as f:
        label_map = json.load(f)
    
    # 加载预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AemNet(num_classes=51).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 创建测试数据集
    test_dataset = AudioDataset(test_data_dir)
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # 获取单个样本
            waveform, label = test_dataset[idx]
            waveform = waveform.unsqueeze(0).to(device)
            
            # 模型推理
            output = model(waveform)
            probabilities = F.softmax(output, dim=0)
            prediction = torch.argmax(output, dim=0)
            confidence = torch.max(probabilities, dim=0)[0]
            
            # 打印结果
            filename = os.path.basename(test_dataset.audio_files[idx])
            print(f'文件名: {filename}')
            print(f'预测类别编号: {prediction.item()}')
            print(f'真实类别编号: {label}')
            print(f'置信度: {confidence.item():.4f}')
            print('-' * 50)

if __name__ == '__main__':
    # 设置模型路径和测试数据目录
    model_path = 'best_pth/ESC-51/full_basic_8.2random/best_model_epoch94_77.45.pth'
    test_data_dir = 'audioSet/test_data'
    
    # 运行推理
    inference(model_path, test_data_dir)