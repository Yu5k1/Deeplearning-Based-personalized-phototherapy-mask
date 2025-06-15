import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from PIL import Image
import os

# 设置类别标签（确保顺序与训练时一致）
dataset_dir = "faceDataset/Train"
folder_names = sorted(os.listdir(dataset_dir))
label_map = {idx: folder for idx, folder in enumerate(folder_names)}

# 预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载模型
model = create_model("vit_base_patch16_224", pretrained=False, num_classes=len(folder_names))
model.load_state_dict(torch.load("best_model_fold_1.pth", map_location=torch.device("cpu")))
model.eval()


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # 添加 batch 维度

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    predicted_label = label_map[predicted.item()]
    print(f"Predicted Class: {predicted_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="ro.jpg")
    args = parser.parse_args()
    predict_image(args.image)
