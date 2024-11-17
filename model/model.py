import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.hub
from typing import List

class DPTDepthHead(nn.Module):
    """
    Depth head inspired by DPT architecture to refine features and produce depth map.
    """
    def __init__(self, in_channels):
        super(DPTDepthHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # Use nn.Identity() if negative depths are acceptable
        )

    def forward(self, x):
        return self.head(x)



class SimpleFusionBlock(nn.Module):
    """
    A simple fusion block that combines features from different sources
    using convolutional layers without channel attention.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the SimpleFusionBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(SimpleFusionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        return self.conv(x)


class CombinedDepthModel(nn.Module):
    """
    Combined model integrating ZoeDepth and DeepLabV3 with DPT depth head.
    """
    def __init__(
        self,
        zoe_repo: str = "isl-org/ZoeDepth",
        zoe_model_name: str = "ZoeD_N",
        fusion_out_channels: int = 256,  # Adjusted to match DPT features
    ):
        super(CombinedDepthModel, self).__init__()

        # Load ZoeDepth pre-trained model
        self.model_zoe = torch.hub.load(zoe_repo, zoe_model_name, pretrained=True)

        # Load DeepLabV3 model for semantic segmentation with ResNet-50 backbone
        segmentation_model = deeplabv3_resnet50(pretrained=True)
        self.segmentation_backbone = segmentation_model.backbone

        # Define channel dimensions
        seg_channels = 2048  # Output channels from ResNet-50 backbone
        zoe_channels = 1     # ZoeDepth outputs a single depth channel
        combined_channels = seg_channels + zoe_channels

        # Define fusion block
        self.fusion_block = nn.Sequential(
            nn.Conv2d(combined_channels, fusion_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_out_channels),
            nn.ReLU(inplace=True)
        )

        # Use DPT Depth Head
        self.depth_head = DPTDepthHead(fusion_out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract depth features from ZoeDepth
        features_zoe = self._extract_depth_features(x)  # Shape: (B, 1, H, W)

        # Extract segmentation features from DeepLabV3 backbone
        features_seg = self._extract_segmentation_features(x)  # Shape: (B, 2048, H/8, W/8)

        # Resize segmentation features to match depth feature spatial dimensions
        features_seg_resized = self._resize_features(features_seg, features_zoe.shape[2:])

        # Combine depth and segmentation features
        combined_features = torch.cat((features_zoe, features_seg_resized), dim=1)  # Shape: (B, combined_channels, H, W)
        fused_features = self.fusion_block(combined_features)  # Shape: (B, fusion_out_channels, H, W)

        # Predict depth map using DPT depth head
        depth_map = self.depth_head(fused_features)  # Shape: (B, 1, H, W)

        return depth_map

    def _extract_depth_features(self, x: torch.Tensor) -> torch.Tensor:
        features_zoe = self.model_zoe.infer(x)  # Expected shape: (B, 1, H, W)
        if features_zoe.dim() == 3:
            features_zoe = features_zoe.unsqueeze(1)
        return features_zoe

    def _extract_segmentation_features(self, x: torch.Tensor) -> torch.Tensor:
        features_seg = self.segmentation_backbone(x)['out']  # Shape: (B, 2048, H/8, W/8)
        return features_seg

    def _resize_features(self, features: torch.Tensor, target_size: tuple) -> torch.Tensor:
        return F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)

    def unfreeze_backbone_layers(self, layers_to_unfreeze: List[str]) -> None:
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

    def freeze_backbone_layers(self, layers_to_freeze: List[str]) -> None:
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

