# vit超参数配置文件
vit_config = {
    # 数据相关
    'data_dir': "../data/cache_tensor",  # 👈 处理后的数据目录 "../data/cache_tensor"
    'img_size': [224, 224],
    'batch_size': 256,

    # 模型相关
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'patch_size': 16,
    'd_ff': 1024,

    # 训练相关
    'num_epochs': 20,
    'learning_rate': 0.0001,   # 学习率
    'weight_decay': 0.05,    # 权重衰减,用于防止过拟合,数值越大,衰减越大

    # 其他
    'checkpoint_dir': "../checkpoint/vit"
}

# Set the configurations
ResNet18_config = {
        # 数据相关
    'data_dir': "../data/cache_tensor",  # 👈 处理后的数据目录
    'img_size': [224, 224],
    'batch_size': 128,

    # 训练相关
    'num_epochs': 20,
    'learning_rate': 0.0001,
    'weight_decay': 0.05,

    # 其他
    'checkpoint_dir': "../checkpoint/resnet18"
}