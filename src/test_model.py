import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from models.vit import VisionTransformer, PatchEmbeddings, LearnedPositionalEmbeddings, ClassificationHead
from models.transformerlayer import TransformerLayer
from models.mha import MultiHeadAttention
from models.ffn import FeedForward
from config import config

def load_model(model_path, num_classes):
    """
    加载训练好的模型
    """
    # 创建模型组件
    patch_emb = PatchEmbeddings(d_model=config['d_model'], patch_size=config['patch_size'], in_channels=3)
    pos_emb = LearnedPositionalEmbeddings(d_model=config['d_model'], max_len=5000)
    mha = MultiHeadAttention(heads=config['n_heads'], d_model=config['d_model'])
    ffn = FeedForward(d_model=config['d_model'], d_ff=config['d_ff'])
    transformer_layer = TransformerLayer(d_model=config['d_model'], self_attn=mha, feed_forward=ffn, dropout_prob=0.1)
    classification_head = ClassificationHead(d_model=config['d_model'], n_hidden=config['d_model'], n_classes=num_classes)
    
    model = VisionTransformer(
        transformer_layer=transformer_layer,
        n_layers=config['n_layers'],
        patch_emb=patch_emb,
        pos_emb=pos_emb,
        classification=classification_head
    )
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def get_test_dataloader(data_dir, img_size=None, batch_size=None):
    """
    从original_data创建测试数据加载器
    """
    if img_size is None:
        img_size = config['img_size']
    if batch_size is None:
        batch_size = config['batch_size']
    
    transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return dataloader, dataset.classes

def test_model(model, dataloader, device):
    """
    测试模型并返回预测结果和真实标签
    """
    model.to(device)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据目录
    original_data_dir = "../data/original_data"
    model_path = "checkpoint/best_vit_model.pth"
    
    # 检查文件是否存在
    if not os.path.exists(original_data_dir):
        print(f"错误: 数据目录不存在: {original_data_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 加载数据
    print("加载测试数据...")
    test_loader, class_names = get_test_dataloader(original_data_dir)
    num_classes = len(class_names)
    print(f"类别数: {num_classes}")
    print(f"类别: {class_names}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(model_path, num_classes)
    
    # 测试模型
    print("开始测试...")
    preds, labels = test_model(model, test_loader, device)
    
    # 计算准确率
    accuracy = np.mean(preds == labels)
    print(".4f")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=class_names))
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png")

if __name__ == "__main__":
    main()