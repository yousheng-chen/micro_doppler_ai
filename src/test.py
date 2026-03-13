from models.vit import VisionTransformer, PatchEmbeddings, LearnedPositionalEmbeddings, ClassificationHead
from models.transformerlayer import TransformerLayer
from models.mha import MultiHeadAttention
from models.ffn import FeedForward

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def get_dataloader(data_dir, img_size: list = [224, 224], batch_size: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((img_size[0], img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, dataset.classes


def build_vit_model(d_model: int = 512, n_heads: int = 8, n_layers: int = 12, 
                   patch_size: int = 16, n_classes: int = 6, d_ff: int = 2048):
    """
    构建完整的Vision Transformer模型
    
    Args:
        d_model: 隐藏维度
        n_heads: 多头注意力头数
        n_layers: Transformer层数
        patch_size: patch大小
        n_classes: 分类类别数
        d_ff: 前馈网络中间维度
    """
    # 1. Patch Embeddings
    patch_emb = PatchEmbeddings(d_model=d_model, patch_size=patch_size, in_channels=3)
    
    # 2. Positional Embeddings
    pos_emb = LearnedPositionalEmbeddings(d_model=d_model, max_len=5000)
    
    # 3. Transformer Layer (用于克隆)
    mha = MultiHeadAttention(heads=n_heads, d_model=d_model)
    ffn = FeedForward(d_model=d_model, d_ff=d_ff)
    transformer_layer = TransformerLayer(d_model=d_model, self_attn=mha, feed_forward=ffn, dropout_prob=0.1)
    
    # 4. Classification Head
    classification_head = ClassificationHead(d_model=d_model, n_hidden=d_model, n_classes=n_classes)
    
    # 5. Vision Transformer
    model = VisionTransformer(
        transformer_layer=transformer_layer,
        n_layers=n_layers,
        patch_emb=patch_emb,
        pos_emb=pos_emb,
        classification=classification_head
    )
    
    return model


def test_model(model, test_loader, criterion, device):
    """测试模型并返回详细结果"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失函数
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率
    ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据目录
    data_dir = "../data/processed_data"
    
    # 加载数据
    print("加载数据集...")
    _, _, test_loader, class_names = get_dataloader(data_dir, img_size=[864, 656], batch_size=4)
    print(f"类别: {class_names}")
    n_classes = len(class_names)
    
    # 构建模型
    print("构建ViT模型...")
    model = build_vit_model(
        d_model=256,
        n_heads=8,
        n_layers=6,
        patch_size=16,
        n_classes=n_classes,
        d_ff=1024
    )
    model = model.to(device)
    
    # 加载训练好的模型
    checkpoint_path = "checkpoint/best_vit_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"加载模型权重: {checkpoint_path}")
    else:
        print(f"未找到模型权重文件: {checkpoint_path}")
        return
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 测试模型
    print("测试模型...")
    test_loss, test_acc, preds, labels = test_model(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试精度: {test_acc:.2f}%")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    print("\n混淆矩阵:")
    print(cm)
    
    # 加载训练历史
    history_path = "checkpoint/training_history.pkl"
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        print(f"加载训练历史: {history_path}")
        
        # 绘制训练历史
        plot_training_history(history, save_path="checkpoint/training_history.png")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(cm, class_names, save_path="checkpoint/confusion_matrix.png")
    else:
        print(f"未找到训练历史文件: {history_path}")


if __name__ == "__main__":
    main()