import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from timm import create_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import numpy as np

# 数据集路径
dataset_dir = "faceDataset/Train"

# 获取类别映射
folder_names = sorted(os.listdir(dataset_dir))
label_map = {folder: idx for idx, folder in enumerate(folder_names)}

# 数据增强（更强的数据增强）
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # 增加旋转角度
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # 更强的颜色抖动
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.75, 1.25)),  # 更强的仿射变换
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 更激进的裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.7, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0),  # 增强RandomErasing
])

# 验证集变换（保持简单但一致）
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.root_dir = root_dir
        self.label_map = label_map
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                label = label_map[folder]
                for image_file in os.listdir(folder_path):
                    if image_file.endswith((".jpg", ".png")):
                        image_path = os.path.join(folder_path, image_file)
                        self.image_paths.append(image_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# 训练函数（加入早停和标签平滑）
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=30, fold=0, patience=5):
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = f"best_model_fold_{fold}.pth"

    for epoch in range(num_epochs):
        print("=" * 50)
        print(f" 第 {epoch + 1}/{num_epochs} 轮训练开始...")
        model.train()

        running_loss, correct_preds, total_preds = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_preds += (outputs.argmax(dim=1) == labels).sum().item()
            total_preds += labels.size(0)

            if batch_idx % 10 == 0:
                print(f" [批次 {batch_idx}/{len(train_loader)}] 训练中: Loss={loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds
        print(f"✅ 训练结束: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")

        # 验证模型
        model.eval()
        val_loss, val_correct_preds, val_total_preds = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader, start=1):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                val_correct_preds += (outputs.argmax(dim=1) == labels).sum().item()
                val_total_preds += labels.size(0)

                if batch_idx % 5 == 0:
                    print(f" [批次 {batch_idx}/{len(val_loader)}] 验证中: Loss={loss.item():.4f}")

        val_loss /= len(val_loader)
        val_accuracy = val_correct_preds / val_total_preds
        print(f" 验证结束: Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")

        # 调整学习率
        scheduler.step()

        # 保存最佳模型并检查早停
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ 最佳模型已保存 (Accuracy: {val_accuracy:.4f})")
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1
            print(f"耐心计数器: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"早停触发，训练提前结束于第 {epoch + 1} 轮")
                break

        print("=" * 50)

    print(f" 训练完成! 最佳验证集准确率: {best_val_acc:.4f}")
    return best_val_acc

# Windows 上需要加这段
if __name__ == "__main__":
    # 创建数据集
    dataset = CustomDataset(root_dir=dataset_dir, label_map=label_map, transform=transform_train)
    labels = np.array(dataset.labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_accs = []

    for fold, (train_index, val_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"折 {fold+1}")
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(CustomDataset(dataset_dir, label_map, transform=transform_val), val_index)

        # Windows 下 num_workers 设为 0
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

        # 创建 ViT 模型
        model = create_model(
        "vit_base_patch16_224", 
        pretrained=True, 
        num_classes=len(folder_names),  
        pretrained_cfg_overlay=dict(file="pytorch_model.bin")  # 替换为实际路径
        )
        model.head = nn.Sequential(
            nn.Dropout(0.7),  # 增加Dropout
            nn.Linear(model.num_features, len(folder_names))
        )

        # 设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 优化器 & 损失函数（加入标签平滑）
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=5e-2)  # 更小的学习率和更大的weight_decay
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
        scheduler = CosineAnnealingLR(optimizer, T_max=15)  # 缩短周期

        # 开始训练（加入早停）
        val_acc = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=30, fold=fold, patience=5)
        val_accs.append(val_acc)

    print(f"交叉验证完成，每个折的验证集准确率：{val_accs}")
    best_fold = np.argmax(val_accs)
    print(f"在折 {best_fold+1} 达到最高验证准确率，准确率为 {val_accs[best_fold]}")