import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model_residual import AemNetResidual as AemNet

def inference(model_path, test_data_dir, batch_size=32):
    # 加载预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AemNet(num_classes=51).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 创建测试数据集和数据加载器
    test_dataset = AudioDataset(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (waveforms, _) in enumerate(test_loader):
            # 将数据移到设备上
            waveforms = waveforms.to(device)
            
            # 模型推理
            outputs = model(waveforms)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidences, _ = torch.max(probabilities, dim=1)
            
            # 打印结果
            for i in range(waveforms.size(0)):
                idx = batch_idx * batch_size + i
                if idx >= len(test_dataset):
                    break
                    
                filename = os.path.basename(test_dataset.audio_files[idx])
                pred_class = predictions[i].item()
                confidence = confidences[i].item()
                
                print(f'文件名: {filename}')
                print(f'预测类别: {pred_class}')
                print(f'置信度: {confidence:.4f}')
                print('-' * 50)

if __name__ == '__main__':
    # 设置模型路径和测试数据目录
    model_path = 'best_pth/ESC-51/2v1_3s/best_model_epoch188_avg95.88.pth'
    test_data_dir = 'audioSet/audio_data'
    
    # 运行推理
    inference(model_path, test_data_dir)