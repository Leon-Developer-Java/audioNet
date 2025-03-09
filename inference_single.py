import os
import torch
import torchaudio
import torch.nn.functional as F
from basic_processor import BasicProcessor
from dataset import AudioDataset
from model_residual import AemNetResidual as AemNet
from audio_augmentation import AugmentationProcessor
from shape_processor import ShapeProcessor

def inference_single(model_path, audio_file, only_shape=False):
    # 加载预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AemNet(num_classes=51).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 创建数据集实例（仅用于处理单个文件）
    audio_dir = os.path.dirname(os.path.abspath(audio_file))
    dataset = AudioDataset(audio_dir)
    
    # 直接使用dataset加载和处理音频数据
    waveform, _ = dataset[0]
    
    # 添加batch维度并移到设备上
    waveform = waveform.unsqueeze(0).to(device)
    print(f'音频数据shape: {waveform.shape}  # [batch_size, channels, time_steps]')
    
    with torch.no_grad():
        # 模型推理
        outputs = model(waveform)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)
        
        # 获取结果
        pred_class = prediction.item()
        conf_value = confidence.item()
        
        # 打印结果
        print(f'\n音频文件: {os.path.basename(audio_file)}')
        print(f'预测类别: {pred_class}')
        print(f'置信度: {conf_value:.4f}')

if __name__ == '__main__':
    # 设置模型路径和音频文件路径
    model_path = 'best_pth/ESC-51/full_basic_8.2random/best_model_epoch94_77.45.pth'
    audio_file = 'audioSet/test_data/50_ignition_sounds_041_5.0s.wav'  # 替换为实际的测试音频文件路径
    
    # 运行推理
    inference_single(model_path, audio_file)