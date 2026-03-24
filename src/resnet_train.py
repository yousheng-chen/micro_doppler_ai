import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models.resnet import ResNet18
from config import ResNet18_config
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import random
from typing import Tuple



# Prepare dataloader
def get_dataloader(data_dir, img_size: list = None, batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if img_size is None:
        img_size = ResNet18_config['img_size']
    if batch_size is None:
        batch_size = ResNet18_config['batch_size']
    
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


def build_resnet18(img_channels: int = 3, first_kernel_size: int = 7, num_classes: int = 6) -> ResNet18:
    model = ResNet18(img_channels = img_channels, first_kernel_size = first_kernel_size, n_classes=num_classes)
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # forward 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # backward and optimize
        loss.backward()
        optimizer.step()
        
        # loss.item(), running_loss是所有loss的累加
        running_loss += loss.item() * inputs.size(0)
        # torch.max(outputs.data, 1) 是对模型的输出进行argmax操作，predicted是预测结果
        _, predicted = torch.max(outputs.data, 1)
        # total 是所有批次的样本数量，correct 是预测正确的样本数量
        total += labels.size(0)
        # (predicted == labels).sum().item() 是当前批次预测正确的样本数量
        correct += (predicted == labels).sum().item()
    
    # avg loss and accuracy for the epoch
    epoch_loss = running_loss / total
    epoch_accuracy = 100 *correct / total
    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


def train_resnet18(num_epochs: int = None, batch_size: int = None, learning_rate: float = None):
    if num_epochs is None:
        num_epochs = ResNet18_config['num_epochs']
    if batch_size is None:
        batch_size = ResNet18_config['batch_size']    
    if learning_rate is None:
        learning_rate = ResNet18_config['learning_rate']
        
    # setting random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)
    
    # clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # loading data
    train_loader, val_loader, test_loader, class_names = get_dataloader(
        data_dir=ResNet18_config['data_dir'], 
        img_size=ResNet18_config['img_size'], 
        batch_size=batch_size
    )
    
    print(f"类别: {class_names}")
    n_classes = len(class_names)
    
    # 构建模型
    print("构建ResNet18模型...")
    model = build_resnet18(img_channels=3, first_kernel_size=7, num_classes=n_classes)
    model = model.to(device)
    
    # print model info
    print(f"模型信息:\n{model.info()}") 
    
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=ResNet18_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 创建checkpoint目录
    checkpoint_dir = ResNet18_config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # initialize lists to store training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # training loop
    best_val_acc = 0.0
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n第 {epoch+1}/{num_epochs} 个Epoch")
        
        # train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # validate on validation set
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # record validation history, training history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # learning rate scheduler step
        scheduler.step()
        
        # print epoch results
        print(f"训练 Loss: {train_loss:.4f}, 训练 Accuracy: {train_acc:.4f}")
        print(f"验证 Loss: {val_loss:.4f}, 验证 Accuracy: {val_acc:.4f}")
        
        # save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, "best_vit_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"保存最好的模型到 {checkpoint_path}")
        
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
    
    with open(os.path.join(checkpoint_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"保存训练历史到 {os.path.join(checkpoint_dir, 'training_history.pkl')}")
    
    # test the best model on the test set
    print("\n测试模型...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_vit_model.pth")))


    # 加载训练历史并绘制
    with open(os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.pkl'), 'rb') as f:
        history = pickle.load(f)
    plot_training_history(history, save_path=os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.png'))
    
    # 加载最好的模型
    model.load_state_dict(torch.load(os.path.join(ResNet18_config['checkpoint_dir'], "best_vit_model.pth")))

    # 计算预测和标签
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = get_predictions_and_labels(model, test_loader, device)   
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(ResNet18_config['checkpoint_dir'], 'confusion_matrix.png'))
    
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
    model = train_resnet18()
    
    
    