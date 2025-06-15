import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from comminute import crop_face_regions
from main import control_all_regions_parallel
import asyncio
import comtypes
import threading

# Load ViT model (using timm)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=5)

# Modify classification head to match training structure
model.head = nn.Sequential(
    nn.Identity(),
    nn.Linear(model.head.in_features, 5)
)

# Load parameters
model.load_state_dict(torch.load("best_model_fold_1.pth", map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing (for ViT input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Light therapy mapping table (5 classes: acne, eczema, herpes, rosacea, ringworm)
light_therapy_map = {
    0: [1, 0, 1],  # Acne: red + blue light
    1: [0, 1, 0],  # Eczema: green light
    2: [1, 0, 1],  # Herpes: red + blue light
    3: [1, 1, 0],  # Rosacea: red + green light
    4: [0, 1, 1],  # Ringworm: blue + green light
}

# Classification function
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Main processing function
def process_image(image_path):
    cropped_images = crop_face_regions(image_path)
    
    region_names = []
    disease_classes = []
    light_therapy_results = []

    for region_name, cropped_path in cropped_images.items():
        if cropped_path is None:
            print(f"Skipping {region_name}: no available image")
            region_names.append(region_name)
            disease_classes.append(-1)
            light_therapy_results.append([0, 0, 0])
            continue
        
        disease_class = classify_image(cropped_path)
        
        if disease_class == 3 and region_name != "nose_upper_cheeks":
            disease_class = 0
        
        region_names.append(region_name)
        disease_classes.append(disease_class)
        light_therapy = light_therapy_map[disease_class]
        light_therapy_results.append(light_therapy)

    result_2x5 = np.array([
        region_names,
        disease_classes
    ])

    result_4x5 = np.array([
        ["1"] + light_therapy_results[0],
        ["2"] + light_therapy_results[1],
        ["3"] + light_therapy_results[2],
        ["4"] + light_therapy_results[3],
        ["5"] + light_therapy_results[4],
    ])

    return result_2x5, result_4x5

# Async wrapper to ensure MTA thread
def run_in_mta_thread(coro):
    def mta_thread():
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)  # Set MTA threading model
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
            comtypes.CoUninitialize()

    thread = threading.Thread(target=mta_thread)
    thread.start()
    thread.join()

# Run full pipeline
if __name__ == "__main__":
    image_path = "face_test.jpg"
    result_2x5, result_4x5 = process_image(image_path)
    
    print("2x5 array (Region, Disease Class):")
    print(result_2x5)
    print("\n4x5 matrix (Region, Red, Green, Blue):")
    print(result_4x5)

    matrix = result_4x5.tolist()
    total_time = 24

    # Run Bluetooth operation in MTA thread
    run_in_mta_thread(control_all_regions_parallel(matrix, total_time))
