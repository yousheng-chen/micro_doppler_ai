from models.vit import VisionTransformer, PatchEmbeddings, LearnedPositionalEmbeddings, ClassificationHead
from models.transformerlayer import TransformerLayer
from models.mha import MultiHeadAttention
from models.ffn import FeedForward
from config import vit_config

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import random

def get_dataloader(data_dir, img_size: list = None, batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if img_size is None:
        img_size = vit_config['img_size']
    if batch_size is None:
        batch_size = vit_config['batch_size']
    
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


def build_vit_model(d_model: int = None, n_heads: int = None, n_layers: int = None, 
                   patch_size: int = None, n_classes: int = None, d_ff: int = None):
    if d_model is None:
        d_model = vit_config['d_model']
    if n_heads is None:
        n_heads = vit_config['n_heads']
    if n_layers is None:
        n_layers = vit_config['n_layers']
    if patch_size is None:
        patch_size = vit_config['patch_size']
    if d_ff is None:
        d_ff = vit_config['d_ff']
    
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


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_vit(num_epochs: int = None, batch_size: int = None, learning_rate: float = None):
    if num_epochs is None:
        num_epochs = vit_config['num_epochs']
    if batch_size is None:
        batch_size = vit_config['batch_size']
    if learning_rate is None:
        learning_rate = vit_config['learning_rate']
    
    """
    训练ViT模型
    """
    # 设置随机种子以确保可重复性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("加载数据集...")
    train_loader, val_loader, test_loader, class_names = get_dataloader(
        vit_config['data_dir'], 
        img_size=vit_config['img_size'], 
        batch_size=batch_size
    )
    print(f"类别: {class_names}")
    n_classes = len(class_names)
    
    # 构建模型
    print("构建ViT模型...")
    model = build_vit_model(
        d_model=vit_config['d_model'],
        n_heads=vit_config['n_heads'],
        n_layers=vit_config['n_layers'],
        patch_size=vit_config['patch_size'],
        n_classes=n_classes,
        d_ff=vit_config['d_ff']
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=vit_config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 创建checkpoint目录
    checkpoint_dir = vit_config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 训练循环
    best_val_acc = 0.0
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n第 {epoch+1}/{num_epochs} 个Epoch")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step()
        
        print(f"训练损失: {train_loss:.4f}, 训练精度: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证精度: {val_acc:.2f}%")
        
        # 保存最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, "best_vit_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"保存最好的模型到 {checkpoint_path}")
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {training_time:.2f}秒")
    print(f"最佳验证精度: {best_val_acc:.2f}%")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    with open(os.path.join(checkpoint_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"保存训练历史到 {os.path.join(checkpoint_dir, 'training_history.pkl')}")
    
    # 测试
    print("\n测试模型...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_vit_model.pth")))

        
    # 加载训练历史并绘制
    with open(os.path.join(vit_config['checkpoint_dir'], 'training_history.pkl'), 'rb') as f:
        history = pickle.load(f)
    plot_training_history(history, save_path=os.path.join(vit_config['checkpoint_dir'], 'training_history.png'))
    
    # 加载最好的模型
    model.load_state_dict(torch.load(os.path.join(vit_config['checkpoint_dir'], "best_vit_model.pth")))
    
    
    # 计算预测和标签
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = get_predictions_and_labels(model, test_loader, device)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(vit_config['checkpoint_dir'], 'confusion_matrix.png'))
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试精度: {test_acc:.2f}%")
    
    
    return model


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


def get_predictions_and_labels(model, test_loader, device):
    """获取模型在测试集上的预测和真实标签"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

if __name__ == "__main__":
    # 训练模型
    print(f"模型信息:\n{model.info()}")
    model = train_vit()
    