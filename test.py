import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import accuracy_score
import os

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建模型
model = create_model(
        "vit_base_patch16_224", 
        pretrained=True, 
        num_classes=5,  
        pretrained_cfg_overlay=dict(file="pytorch_model.bin")  # 替换为实际路径
        )
model.head = nn.Sequential(
    nn.Dropout(0.3),  # 加入 Dropout 层（30% 随机失活）
    nn.Linear(model.num_features, 5)  # 输出层，修改为实际类别数
)

# 载入保存的模型权重
model.load_state_dict(torch.load("best_model_fold_4.pth", map_location=torch.device("cpu")), strict=False)
model.eval()  # 设置模型为评估模式

# 创建测试集数据集和数据加载器
test_dir = "faceDataset/Test"  # 测试集路径
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 计算测试集准确率
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算精度
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
