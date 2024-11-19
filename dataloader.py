import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random

def remove_leading_slash(s):
    return s.lstrip('/\\')

class DataLoadPreprocess(Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        self.transform = transform

        # Paths to data (Update these paths according to your dataset location)
        base_path = "/content/project1/train"  # Base directory for your data

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
            # Geometric transformations
            self.geometric_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0)),
            ])
            # Color transformations
            self.color_transforms = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            )
        else:
            self.geometric_transforms = None
            self.color_transforms = None

        # Define transforms for image and depth
        self.image_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Chuẩn hóa theo ImageNet
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

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
            # Use the same seed for both image and depth_gt to apply the same geometric transformations
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            if self.geometric_transforms is not None:
                image = self.geometric_transforms(image)
            random.seed(seed)
            torch.manual_seed(seed)
            if self.geometric_transforms is not None:
                depth_gt = self.geometric_transforms(depth_gt)

            # Apply color jitter only to the image
            if self.color_transforms is not None:
                image = self.color_transforms(image)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            depth_gt = self.depth_transform(depth_gt)
        else:
            # Nếu không có transform, chuyển đổi thành tensor và chuẩn hóa
            image = self.image_transform(image)
            depth_gt = self.depth_transform(depth_gt)

        depth_gt = depth_gt.squeeze(0) / 1000.0  # Convert from mm to meters

        # Create mask
        mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        mask = mask.float()  # Không cần unsqueeze nếu đã có chiều phù hợp

        sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.filenames)
