import os
import torch
from PIL import Image
from tqdm import tqdm

# ======================
# 配置
# ======================
data_dir = "../data/all_data"              # 👈 数据目录
save_dir = "../data/cache_tensor"      # 👈 保存目录
img_size = 224                 # 👈 输入尺寸（非常关键）
chunk_size = 1000              # 👈 每个文件多少样本

os.makedirs(save_dir, exist_ok=True)


# ======================
# 构建 file_list
# ======================
def build_file_list(data_dir):
    file_list = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_path):
            path = os.path.join(cls_path, fname)
            file_list.append((path, class_to_idx[cls]))

    return file_list


file_list = build_file_list(data_dir)

print(f"📊 Total images: {len(file_list)}")
print(f"📂 Save dir: {save_dir}")
print("🚀 Start processing...\n")


# ======================
# 主处理
# ======================
data = []
labels = []
chunk_id = 0

for i, (path, label) in enumerate(tqdm(file_list, desc="Processing", ncols=100)):

    try:
        # 读取 + resize
        img = Image.open(path).convert("RGB")
        img = img.resize((img_size, img_size))

        # 👉 转 tensor（核心）
        img = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(img_size, img_size, 3)
             .numpy())
        ).permute(2, 0, 1)   # HWC → CHW

        data.append(img)
        labels.append(label)

    except Exception as e:
        print(f"❌ Error: {path} -> {e}")
        continue

    # ======================
    # 分块保存
    # ======================
    if (i + 1) % chunk_size == 0:
        save_path = os.path.join(save_dir, f"tensor_part_{chunk_id}.pt")

        torch.save((
            torch.stack(data),            # 👈 直接 stack
            torch.tensor(labels)
        ), save_path)

        print(f"\n💾 Saved: {save_path} ({len(data)})")

        data = []
        labels = []
        chunk_id += 1


# ======================
# 保存剩余
# ======================
if len(data) > 0:
    save_path = os.path.join(save_dir, f"tensor_part_{chunk_id}.pt")

    torch.save((
        torch.stack(data),
        torch.tensor(labels)
    ), save_path)

    print(f"\n💾 Saved final: {save_path} ({len(data)})")


print("\n✅ All done!")





"""
🚀 训练时如何用（重点）
👉 直接加载 chunk
data, labels = torch.load("cache_tensor/tensor_part_0.pt")


👉 Dataset 写法（超简单）
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
        
👉 DataLoader
dataset = TensorDataset(data, labels)

loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

"""