import argparse
import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import models, transforms
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Generate Pseudo Semantic Labels using DeepLabV3")
    parser.add_argument('--list_file', type=str, required=True, help='Path to train.txt or test.txt')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save semantic labels')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu)')
    return parser.parse_args()

def generate_semantic_labels(list_file, root_dir, output_dir, device='cuda'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load pre-trained DeepLabV3 model
    semantic_model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
    semantic_model.eval()
    
    # Define transformation
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    with open(list_file, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Generating Semantic Labels"):
        parts = line.strip().split()
        if len(parts) < 2:
            print(f"Skipping invalid line: {line}")
            continue
        img_path, depth_path = parts[0], parts[1]
        full_img_path = os.path.join(root_dir, img_path)
        
        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            continue
        
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = semantic_model(input_tensor)['out'][0]
        semantic_pred = output.argmax(0).cpu().numpy().astype(np.uint8)
        
        # Lưu semantic_pred dưới dạng PNG
        semantic_image = Image.fromarray(semantic_pred)
        semantic_filename = img_path.replace('.jpg', '.png').replace('.png', '_semantic.png')
        semantic_save_path = os.path.join(output_dir, semantic_filename)
        
        # Tạo các thư mục con nếu cần
        semantic_save_dir = os.path.dirname(semantic_save_path)
        if not os.path.exists(semantic_save_dir):
            os.makedirs(semantic_save_dir)
        
        semantic_image.save(semantic_save_path)

if __name__ == '__main__':
    args = get_args()
    generate_semantic_labels(args.list_file, args.root_dir, args.output_dir, args.device)