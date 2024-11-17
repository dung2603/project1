# dataloader.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random

def remove_leading_slash(s):
    return s.lstrip('/\\')

class DataLoadPreprocess(Dataset):
    def __init__(self, mode, **kwargs):
        self.mode = mode

        # Define image transform (including data augmentation)
        self.image_transform = transforms.Compose([
            transforms.Pad((0, 0, 14 - (640 % 14), 14 - (480 % 14))),  # Add padding to make dimensions divisible by 14
            transforms.Resize((364, 364)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization theo ImageNet
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Define depth transform
        self.depth_transform = transforms.Compose([
            transforms.Pad((0, 0, 14 - (640 % 14), 14 - (480 % 14))),
            transforms.Resize((364, 364), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Define data augmentation transforms
        if mode == 'train':
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop(size=364, scale=(0.8, 1.0)),
            ])
        else:
            self.augmentation = None  # Không áp dụng augmentation cho validation/test

        # Paths to data (Update these paths according to your dataset location)
        base_path = r'C:\filemohinh\modelmoi\train'  # Base directory for your data

        if mode == 'train':
            self.data_path = base_path  # Path to training images and depths
            self.filenames_file = os.path.join(base_path, 'nyudepthv2_train_files_with_gt.txt')  # Path to your train.txt file
        else:
            self.data_path = base_path  # Path to test images and depths
            self.filenames_file = os.path.join(base_path, 'nyudepthv2_test_files_with_gt.txt')  # Path to your test.txt file

        # Read the list of filenames
        with open(self.filenames_file, 'r') as f:
            self.filenames = f.readlines()

        # Depth limits
        self.min_depth = 0.1
        self.max_depth = 10.0

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
        if self.mode == 'train':
            # Apply the same transformation to both image and depth
            # Random Horizontal Flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                depth_gt = TF.hflip(depth_gt)

            # Random Rotation
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            depth_gt = TF.rotate(depth_gt, angle, interpolation=InterpolationMode.NEAREST)

            # Random Resized Crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            image = TF.resized_crop(image, i, j, h, w, size=(364, 364), interpolation=InterpolationMode.BILINEAR)
            depth_gt = TF.resized_crop(depth_gt, i, j, h, w, size=(364, 364), interpolation=InterpolationMode.NEAREST)

            # Color Jitter (applied only to image)
            image = self.augmentation(image)

        # Apply transforms
        image = self.image_transform(image)
        depth_gt = self.depth_transform(depth_gt).squeeze(0) / 1000.0  # Convert from mm to meters

        # Create mask
        mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        mask = mask.float().unsqueeze(0)  # Add channel dimension

        # If segmentation labels are available, load them
        # Assuming that segmentation labels are in the 4th column of sample_items
        if len(sample_items) >= 4:
            seg_rel_path = sample_items[3]
            seg_path = os.path.join(self.data_path, remove_leading_slash(seg_rel_path))
            seg_gt = Image.open(seg_path)
            seg_gt = transforms.Resize((364, 364), interpolation=InterpolationMode.NEAREST)(seg_gt)
            seg_gt = transforms.ToTensor()(seg_gt).long().squeeze(0)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask, 'segmentation': seg_gt}
        else:
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.filenames)


# If you want to test the dataset
if __name__ == '__main__':
    dataset = DataLoadPreprocess(mode='train')

    # Get one sample
    sample = dataset[0]

    print("Image shape:", sample['image'].shape)
    print("Depth shape:", sample['depth'].shape)
    print("Focal length:", sample['focal'])
    print("Mask shape:", sample['mask'].shape)
    if 'segmentation' in sample:
        print("Segmentation shape:", sample['segmentation'].shape)
