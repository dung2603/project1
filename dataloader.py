# dataloader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random

def remove_leading_slash(s):
    return s.lstrip('/\\')

class DataLoadPreprocess(Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        self.transform = transform

        # Paths to data (Update these paths according to your dataset location)
        base_path = r'C:\filemohinh\modelmoi\train'  # Base directory for your data

        if mode == 'train':
            self.data_path = base_path
            self.filenames_file = os.path.join(base_path, 'nyudepthv2_train_files_with_gt.txt')
        else:
            self.data_path = base_path
            self.filenames_file = os.path.join(base_path, 'nyudepthv2_test_files_with_gt.txt')

        # Read the list of filenames
        with open(self.filenames_file, 'r') as f:
            self.filenames = f.readlines()

        # Depth limits
        self.min_depth = 0.1
        self.max_depth = 10.0

        # Data augmentation transforms
        if mode == 'train':
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        else:
            self.augmentation = None

    def __getitem__(self, idx):
        sample_line = self.filenames[idx].strip()
        sample_items = sample_line.split()

        if len(sample_items) < 3:
            raise ValueError(f"Invalid line in filename file: {sample_line}")

        image_rel_path = sample_items[0]
        depth_rel_path = sample_items[1]
        focal = torch.tensor(float(sample_items[2]), dtype=torch.float32)

        # Construct full paths
        image_path = os.path.join(self.data_path, remove_leading_slash(image_rel_path))
        depth_path = os.path.join(self.data_path, remove_leading_slash(depth_rel_path))

        # Read image and depth
        image = Image.open(image_path).convert('RGB')
        depth_gt = Image.open(depth_path)

        # Apply data augmentation if in training mode
        if self.mode == 'train' and self.augmentation is not None:
            # Sử dụng cùng một seed cho cả image và depth để áp dụng cùng một phép biến đổi
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.augmentation(image)
            random.seed(seed)
            torch.manual_seed(seed)
            depth_gt = self.augmentation(depth_gt)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            depth_gt = self.transform(depth_gt)
        else:
            # Nếu không có transform, chuyển đổi thành tensor và chuẩn hóa
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)
            depth_gt = transforms.ToTensor()(depth_gt)

        depth_gt = depth_gt.squeeze(0) / 1000.0  # Convert from mm to meters

        # Create mask
        mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        mask = mask.float().unsqueeze(0)  # Add channel dimension

        sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.filenames)

# Nếu bạn muốn kiểm tra dataset
if __name__ == '__main__':
    dataset = DataLoadPreprocess(mode='train')

    # Get one sample
    sample = dataset[0]

    print("Image shape:", sample['image'].shape)
    print("Depth shape:", sample['depth'].shape)
    print("Focal length:", sample['focal'])
    print("Mask shape:", sample['mask'].shape)
