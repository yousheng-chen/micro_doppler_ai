from models.vit import VisionTransformer, PatchEmbeddings, LearnedPositionalEmbeddings, ClassificationHead
from models.transformerlayer import TransformerLayer
from models.mha import MultiHeadAttention
from models.ffn import FeedForward
from config import vit_config
from models.resnet import ResNet18
from config import ResNet18_config
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

from PIL import Image
from torch.utils.data import Dataset


#######################################
from resnet_train import get_tensor_dataloader, build_resnet18
from training import build_vit_model, validate, get_predictions_and_labels, plot_training_history, plot_confusion_matrix

from training import train_epoch as vit_train_epoch
from resnet_train import train_epoch as resnet18_train_epoch


def train():
    # setting random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # loading data
    train_loader, val_loader, test_loader, class_names = get_tensor_dataloader(
        data_dir=ResNet18_config['data_dir'], 
        batch_size=ResNet18_config['batch_size'],
    )
   
    print(f"类别: {class_names}")
    n_classes = len(class_names)
    
    ###########################################################################################
    # 构建构建ResNet18模型模型
    print("构建ResNet18模型...")
    resnet18_model = build_resnet18(img_channels=3, first_kernel_size=7, num_classes=n_classes)
    resnet18_model = resnet18_model.to(device)
    # print model info
    print(f"模型信息:\n{resnet18_model.info()}") 
    
    
    # loss function and optimizer
    resnet18_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    resnet18_optimizer = torch.optim.AdamW(resnet18_model.parameters(), lr=ResNet18_config['learning_rate'], weight_decay=ResNet18_config['weight_decay'])
    resnet18_scheduler = torch.optim.lr_scheduler.StepLR(resnet18_optimizer, step_size=5, gamma=0.1) # 等间隔调整（Step Decay）
    ###########################################################################################
    
    
    ##########################################################################################
    # 构建vit模型
    print("构建ViT模型...")
    vit_model = build_vit_model(
        d_model=vit_config['d_model'],
        n_heads=vit_config['n_heads'],
        n_layers=vit_config['n_layers'],
        patch_size=vit_config['patch_size'],
        n_classes=n_classes,
        d_ff=vit_config['d_ff']
    )
    vit_model = vit_model.to(device)
    # 打印模型信息
    total_params = sum(p.numel() for p in vit_model.parameters())
    trainable_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 损失函数和优化器
    vit_criterion = nn.CrossEntropyLoss()
    vit_optimizer = optim.AdamW(vit_model.parameters(), lr=vit_config['learning_rate'], weight_decay=vit_config['weight_decay'])
    vit_scheduler = optim.lr_scheduler.CosineAnnealingLR(vit_optimizer, T_max=vit_config['num_epochs'])  # 余弦退火学习率调整
    ############################################################################################
    
    
    
    
    
    
    
    # initialize lists to store training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # training loop
    best_val_acc = 0.0
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(ResNet18_config['num_epochs']):
        print(f"\n第 {epoch+1}/{ResNet18_config['num_epochs']} 个Epoch")
        
        # train for one epoch
        train_loss, train_acc = resnet18_train_epoch(resnet18_model, train_loader, resnet18_criterion, resnet18_optimizer, device)
        
        # validate on validation set
        val_loss, val_acc = validate(resnet18_model, val_loader, resnet18_criterion, device)
        
        # record validation history, training history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # learning rate scheduler step
        resnet18_scheduler.step()
        
        # print epoch results
        print(f"训练 Loss: {train_loss:.4f}, 训练 Accuracy: {train_acc:.4f}")
        print(f"验证 Loss: {val_loss:.4f}, 验证 Accuracy: {val_acc:.4f}")
        
        # save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(ResNet18_config['checkpoint_dir'], "best_resnet18_model.pth")
            torch.save(resnet18_model.state_dict(), checkpoint_path)
            print(f"保存最好的模型到 {checkpoint_path}")
        
        torch.cuda.empty_cache()
        
    # print training time    
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"最佳验证精度: {best_val_acc:.2f}%")
        
        # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    
    with open(os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"保存训练历史到 {os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.pkl')}")
    
    # test the best model on the test set
    print("\n测试模型...")
    resnet18_model.load_state_dict(torch.load(os.path.join(ResNet18_config['checkpoint_dir'], "best_resnet18_model.pth")))


    # 加载训练历史并绘制
    with open(os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.pkl'), 'rb') as f:
        history = pickle.load(f)
    plot_training_history(history, save_path=os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.png'))
    
    # 加载最好的模型
    resnet18_model.load_state_dict(torch.load(os.path.join(ResNet18_config['checkpoint_dir'], "best_resnet18_model.pth")))

    # 计算预测和标签
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = get_predictions_and_labels(resnet18_model, test_loader, device)   
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(ResNet18_config['checkpoint_dir'], 'confusion_matrix.png'))
    
    test_loss, test_acc = validate(resnet18_model, test_loader, resnet18_criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试精度: {test_acc:.2f}%")
    

    
    # 清空GPU缓存,训练vit模型
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化历史记录 ViT模型
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 训练循环
    best_val_acc = 0.0
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(vit_config['num_epochs']):
        print(f"\n第 {epoch+1}/{vit_config['num_epochs']} 个Epoch")
        
        # 训练
        train_loss, train_acc = vit_train_epoch(vit_model, train_loader, vit_criterion, vit_optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(vit_model, val_loader, vit_criterion, device)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        vit_scheduler.step()
        
        print(f"训练损失: {train_loss:.4f}, 训练精度: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证精度: {val_acc:.2f}%")
        
        # 保存最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(vit_config['checkpoint_dir'], "best_vit_model.pth")
            torch.save(vit_model.state_dict(), checkpoint_path)
            print(f"保存最好的模型到 {checkpoint_path}")
            
        torch.cuda.empty_cache()
    
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
    with open(os.path.join(vit_config['checkpoint_dir'], 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"保存训练历史到 {os.path.join(vit_config['checkpoint_dir'], 'training_history.pkl')}")
    
    # 测试
    print("\n测试模型...")
    vit_model.load_state_dict(torch.load(os.path.join(vit_config['checkpoint_dir'], "best_vit_model.pth")))

        
    # 加载训练历史并绘制
    with open(os.path.join(vit_config['checkpoint_dir'], 'training_history.pkl'), 'rb') as f:
        history = pickle.load(f)
    plot_training_history(history, save_path=os.path.join(vit_config['checkpoint_dir'], 'training_history.png'))
    
    # 加载最好的模型
    vit_model.load_state_dict(torch.load(os.path.join(vit_config['checkpoint_dir'], "best_vit_model.pth")))
    
    
    # 计算预测和标签
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = get_predictions_and_labels(vit_model, test_loader, device)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(vit_config['checkpoint_dir'], 'confusion_matrix.png'))
    
    test_loss, test_acc = validate(vit_model, test_loader, vit_criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试精度: {test_acc:.2f}%")


if __name__ == "__main__":
    train()

    print("successfully imported all modules")