import torch
import torch.nn as nn
from timm import create_model
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 定义 ViT 模型
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ViTClassifier, self).__init__()
        self.vit = create_model("vit_base_resnet50_384", pretrained=False)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# 加载 .npz 权重
model = ViTClassifier(num_classes=5)
npz_path = "D:/Project/model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz"
weights = np.load(npz_path)
state_dict = {k: torch.from_numpy(v) for k, v in weights.items()}

# 映射权重
new_state_dict = {}
for npz_key, value in state_dict.items():
    if npz_key == "cls_token":
        new_state_dict["cls_token"] = value
    elif npz_key == "pos_embed":
        new_state_dict["pos_embed"] = value
    elif npz_key.startswith("Transformer/encoder_norm"):
        new_key = "norm." + npz_key.split("/")[-1]
        new_state_dict[new_key] = value
    elif "MultiHeadDotProductAttention" in npz_key:
        parts = npz_key.split("/")
        layer_num = parts[1].split("_")[1]
        attn_part = parts[3].split("_")[1]
        new_key = f"blocks.{layer_num}.attn.{attn_part}.{parts[-1]}"
        new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=False)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载训练数据集
train_dataset = datasets.ImageFolder("D:/Project/face_disease/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 训练循环
num_epochs = 5  # 训练 5 轮，可以改多点
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# 保存模型
torch.save(model.state_dict(), "D:/Project/vit_trained.pth")
print("训练完成，模型已保存")