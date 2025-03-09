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
    
    # 创建数据集实例和音频处理器（仅用于处理单个文件）
    audio_dir = os.path.dirname(os.path.abspath(audio_file))
    dataset = AudioDataset(audio_dir)

    # 获取音频文件的索引
    audio_file_path = os.path.abspath(audio_file)
    try:
        file_idx = dataset.audio_files.index(audio_file_path)
    except ValueError:
        print(f"错误：找不到音频文件 {audio_file}")
        print(f"音频目录：{audio_dir}")
        print("可用的音频文件：")
        for file in dataset.audio_files:
            print(f"  - {file}")
        return
    
    # 获取音频数据
    if only_shape:
        # 仅调整形状，不做其他处理
        waveform, sample_rate = torchaudio.load(audio_file_path)
        processor = BasicProcessor()  # 创建BasicProcessor实例
        waveform = processor.apply_basic_processing(waveform)
    else:
        # 使用数据集的处理方式（包含数据增强）
        waveform, _ = dataset[file_idx]
    
    # 添加batch维度并移到设备上
    waveform = waveform.unsqueeze(0).to(device)
    print(f'音频数据shape: {waveform.shape}  # [batch_size, channels, time_steps]')
    with torch.no_grad():
        # 模型推理
        outputs = model(waveform)
        outputs = outputs.unsqueeze(0)  # 添加batch维度
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)
        
        # 获取结果
        pred_class = prediction.item()
        conf_value = confidence.item()
        
        # 打印结果
        print(f'\n音频文件: {os.path.basename(audio_file)}')
        print(f'处理方式: {"仅调整形状" if only_shape else "完整数据增强"}')
        print(f'预测类别: {pred_class}')
        print(f'置信度: {conf_value:.4f}')

if __name__ == '__main__':
    # 设置模型路径和音频文件路径
    model_path = 'best_pth/ESC-51/2v1_3s/best_model_epoch188_avg95.88.pth'
    audio_file = 'audioSet/test_data/50_ignition_sounds_041_5.0s.wav'  # 替换为实际的测试音频文件路径
    
    # 运行推理 - 使用数据增强
    print("\n=== 使用数据增强进行推理 ===")
    inference_single(model_path, audio_file)
    
    # 运行推理 - 仅调整形状
    print("\n=== 仅调整形状进行推理 ===")
    inference_single(model_path, audio_file)