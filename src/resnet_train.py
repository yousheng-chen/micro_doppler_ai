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
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Prepare dataloader
def get_dataloader(data_dir, seed=42, img_size: list = None, batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if img_size is None:
        img_size = ResNet18_config['img_size']
    if batch_size is None:
        batch_size = ResNet18_config['batch_size']
    
    transform = transforms.Compose(
        [
            transforms.Resize((img_size[0], img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]
    )
    

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 打印数据集大小
    print(f"总样本数: {total_size}, 训练样本数: {train_size}, 验证样本数: {val_size}, 测试样本数: {test_size}")
    # # 打印每个类别的样本数量
    # class_counts = {}
    # for _, label in dataset:
    #     class_counts[label] = class_counts.get(label, 0) + 1
    # print("每个类别的样本数量:")
    # for label, count in class_counts.items():
    #     print(f"  类别 {label}: {count}")
    # print(class_counts)
    """
    总样本数: 35999, 训练样本数: 25199, 验证样本数: 5399, 测试样本数: 5401
    
    """    

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True)
    
    return train_loader, val_loader, test_loader, dataset.classes

def get_all_data(datapath) -> Tuple[torch.Tensor, torch.Tensor]:
    # Implementation for getting all data from a given path
    all_data = []
    all_labels = []
    obj = torch.load(path)

    # 2️⃣ 兼容不同保存格式
    if isinstance(obj, dict):
        data = obj["data"]
        labels = obj["labels"]
    else:
        data, labels = obj
        
    all_data.append(data)
    all_labels.append(labels)
    
    dataset = TensorDataset(all_data, all_labels)
    
    # 4️⃣ 划分数据集
    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 5️⃣ DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)


    return train_loader, val_loader, test_loader, dataset.labels.unique().tolist()
        
def get_tensor_dataloader(data_dir, batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    all_data = []
    all_labels = []

    # 1️⃣ 遍历 cache 目录
    for file in os.listdir(data_dir):
        if file.endswith(".pt"):
            path = os.path.join(data_dir, file)
            obj = torch.load(path)

            # 2️⃣ 兼容不同保存格式
            if isinstance(obj, dict):
                data = obj["data"]
                labels = obj["labels"]
            else:
                data, labels = obj

            all_data.append(data)
            all_labels.append(labels)
            print(f"加载 {path} ({len(data)})")
    print(f"\n总样本数: {sum(len(d) for d in all_data)}")
    
    # 3️⃣ 拼接所有块
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    torch.save((all_data, all_labels), "merged.pt")
    
    dataset = TensorDataset(all_data, all_labels)

    for _ in range(5):  # 打乱5次
        # 🔥 1. 打乱索引
        perm = torch.randperm(len(all_data))

        # 🔥 2. 重新排列数据
        all_data = all_data[perm]
        all_labels = all_labels[perm]

    # 🔥 3. 再构建 dataset
    dataset = TensorDataset(all_data, all_labels)

    # 4️⃣ 划分数据集
    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 5️⃣ DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)


    return train_loader, val_loader, test_loader, dataset.labels.unique().tolist()

def build_resnet18(img_channels: int = 3, first_kernel_size: int = 7, num_classes: int = 9) -> ResNet18:
    model = ResNet18(img_channels = img_channels, first_kernel_size = first_kernel_size, n_classes=num_classes)
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device).float()
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
            inputs = inputs.to(device).float()
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
        torch.cuda.manual_seed_all(seed)
    
    # clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # loading data
    train_loader, val_loader, test_loader, class_names = get_tensor_dataloader(
        data_dir=ResNet18_config['data_dir'], 
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=ResNet18_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # 等间隔调整（Step Decay）
    
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
            checkpoint_path = os.path.join(checkpoint_dir, "best_resnet18_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
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
    
    with open(os.path.join(checkpoint_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"保存训练历史到 {os.path.join(checkpoint_dir, 'training_history.pkl')}")
    
    # test the best model on the test set
    print("\n测试模型...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_resnet18_model.pth")))


    # 加载训练历史并绘制
    with open(os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.pkl'), 'rb') as f:
        history = pickle.load(f)
    plot_training_history(history, save_path=os.path.join(ResNet18_config['checkpoint_dir'], 'training_history.png'))
    
    # 加载最好的模型
    model.load_state_dict(torch.load(os.path.join(ResNet18_config['checkpoint_dir'], "best_resnet18_model.pth")))

    # 计算预测和标签
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = get_predictions_and_labels(model, test_loader, device)   
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
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
            images = images.to(device).float()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels    
    
    
if __name__ == "__main__":
    # 训练模型
    model = train_resnet18()
    
    
    