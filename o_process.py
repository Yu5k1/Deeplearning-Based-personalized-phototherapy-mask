# process.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from comminute import crop_face_regions

# 加载 ViT 模型（使用 timm）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=5)  # 5 类：Acne, Eczema, Herpes, Rosacea, TineaVersicolor

# 修改分类头以匹配训练时的结构
model.head = nn.Sequential(
    nn.Identity(),  # 占位，模拟 head.0（可能是一个占位层）
    nn.Linear(model.head.in_features, 5)  # head.1，匹配训练时的分类头（5 类）
)

# 加载参数
model.load_state_dict(torch.load("best_model_fold_1.pth", map_location=device))
model = model.to(device)
model.eval()

# 图像预处理（ViT 输入）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 光疗映射表（5 类：Acne, Eczema, Herpes, Rosacea, TineaVersicolor）
light_therapy_map = {
    0: [1, 0, 1],  # 痤疮 (Acne): 红光+蓝光
    1: [0, 1, 0],  # 湿疮 (Eczema): 绿光
    2: [1, 0, 1],  # 疱疹 (Herpes): 红光+蓝光
    3: [1, 1, 0],  # 酒糟鼻 (Rosacea): 红光+绿光
    4: [0, 1, 1],  # 癣 (TineaVersicolor): 蓝光+绿光
}

# 分类函数
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 主流程
def process_image(image_path):
    # 裁剪面部区域
    cropped_images = crop_face_regions(image_path)
    
    # 存储结果
    region_names = []
    disease_classes = []
    light_therapy_results = []

    # 对每个区域进行分类
    for region_name, cropped_path in cropped_images.items():
        if cropped_path is None:
            print(f"Skipping {region_name}: No image available")
            region_names.append(region_name)
            disease_classes.append(-1)  # 表示未检测到
            light_therapy_results.append([0, 0, 0])
            continue
        
        # 分类
        disease_class = classify_image(cropped_path)
        
        # 酒糟鼻 (Rosacea, 3) 只能出现在 nose_upper_cheeks
        if disease_class == 3 and region_name != "nose_upper_cheeks":
            disease_class = 0  # 转为 Acne
        
        # 存储结果
        region_names.append(region_name)
        disease_classes.append(disease_class)
        
        # 获取光疗方案
        light_therapy = light_therapy_map[disease_class]
        light_therapy_results.append(light_therapy)

    # 构建 2×5 数组（区域名称 + 皮肤病类别）
    result_2x5 = np.array([
        region_names,
        disease_classes
    ])

    # 构建 4×5 矩阵（每行：区域名称, 红光, 黄光, 蓝光）
    result_4x5 = np.array([
        ["1"] + light_therapy_results[0],
        ["2"] + light_therapy_results[1],
        ["3"] + light_therapy_results[2],
        ["4"] + light_therapy_results[3],
        ["5"] + light_therapy_results[4],
    ])

    return result_2x5, result_4x5

# 运行
if __name__ == "__main__":
    image_path = "face_test.jpg"
    result_2x5, result_4x5 = process_image(image_path)
    
    print("2×5 Array (Region, Disease Class):")
    print(result_2x5)
    print("\n4×5 Matrix (Region, Red, Yellow, Blue):")
    print(result_4x5)