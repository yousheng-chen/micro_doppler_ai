# vit超参数配置文件
vit_config = {
    # 数据相关
    'data_dir': "../data/0db",
    'img_size': [224, 224],
    'batch_size': 32,

    # 模型相关
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'patch_size': 16,
    'd_ff': 1024,

    # 训练相关
    'num_epochs': 10,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,

    # 其他
    'checkpoint_dir': "checkpoint"
}

# Set the configurations
ResNet18_config = {
        # 数据相关
    'data_dir': "../data/0db",
    'img_size': [224, 224],
    'batch_size': 32,

    # 训练相关
    'num_epochs': 10,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,

    # 其他
    'checkpoint_dir': "ResNet18_checkpoint"
}