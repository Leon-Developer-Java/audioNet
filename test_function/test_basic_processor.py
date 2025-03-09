import os
import random
import torch
import torchaudio
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from basic_processor import BasicProcessor

def test_basic_processing():
    # 初始化音频处理器
    processor = BasicProcessor()
    
    # 获取音频文件列表
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audioSet', 'audio_data')
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # 随机选择两个音频文件
    selected_files = random.sample(audio_files, 2)
    
    for i, audio_file in enumerate(selected_files):
        # 加载音频
        file_path = os.path.join(audio_dir, audio_file)
        waveform, sample_rate = torchaudio.load(file_path)
        
        print(f'\n处理文件: {audio_file}')
        print(f'原始音频形状: {waveform.shape}')
        print(f'原始采样率: {sample_rate}')
        
        # 应用基础处理
        processed_waveform = processor.apply_basic_processing(waveform)
        
        print(f'处理后音频形状: {processed_waveform.shape}')
        print(f'处理后采样率: {processor.sample_rate}')
        
        # 保存处理后的音频
        output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'processed_audio_{i+1}.wav')
        torchaudio.save(
            output_file,
            processed_waveform,
            processor.sample_rate
        )
        print(f'已保存处理后的音频到: {output_file}')

if __name__ == '__main__':
    test_basic_processing()