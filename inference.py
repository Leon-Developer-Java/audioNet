import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from dataset import AudioDataset
from model_residual import AemNetResidual as AemNet

def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图形
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('混淆矩阵')
    plt.colorbar()
    
    # 设置坐标轴
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path)
    plt.close()

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
    
    # 用于收集预测结果
    all_predictions = []
    all_labels = []
    
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
            
            # 收集结果
            all_predictions.append(prediction.item())
            all_labels.append(label)
            
            # 打印结果
            filename = os.path.basename(test_dataset.audio_files[idx])
            pred_class = label_map[str(prediction.item())]
            true_class = label_map[str(label)]
            print(f'Filename: {filename}')
            print(f'Predicted Class: {pred_class}')
            print(f'True Class: {true_class}')
            print(f'Confidence: {confidence.item():.4f}')
            print('-' * 50)
    
    # 生成混淆矩阵
    unique_labels = sorted(set(test_dataset.label_names))
    class_names = [label_map[str(i)] for i in range(len(unique_labels))]
    plot_confusion_matrix(all_labels, all_predictions, class_names)

if __name__ == '__main__':
    # 设置模型路径和测试数据目录
    model_path = 'best_pth/ESC-51/full_basic_8.2random/best_model_epoch78_75.pth'
    test_data_dir = 'audioSet/test_data'
    
    # 运行推理并生成混淆矩阵
    inference(model_path, test_data_dir)