import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='plots/training_history.png'):
    """Plot training history curves including losses and accuracies"""
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm, classes, save_path='plots/confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_class_distribution(class_counts, class_names, save_path='plots/class_distribution.png'):
    """Plot class distribution in the dataset"""
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(class_counts)), class_counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_learning_rate(learning_rates, save_path='plots/learning_rate.png'):
    """Plot learning rate changes over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(learning_rates) + 1), learning_rates)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()