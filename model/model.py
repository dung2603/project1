import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.hub
from typing import List

# Đảm bảo rằng bạn đã import UNet và các thành phần của nó
from .unet_parts import DoubleConv, Down, Up, OutConv
from .unet_model import UNet

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

class UNetDecoder(nn.Module):
    """
    UNet-based decoder for processing fused features.
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNetDecoder, self).__init__()
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=bilinear)

    def forward(self, x):
        return self.unet(x)

class CombinedDepthModel(nn.Module):
    """
    Combined model that integrates ZoeDepth for depth estimation and
    DeepLabV3 for semantic segmentation to produce an enhanced depth map.
    Utilizes UNet as the decoder after feature fusion.
    """

    def __init__(
        self,
        zoe_repo: str = "isl-org/ZoeDepth",
        zoe_model_name: str = "ZoeD_N",
        fusion_out_channels: int = 128,
        unet_bilinear: bool = False
    ):
        """
        Initializes the CombinedDepthModel.

        Args:
            zoe_repo (str): Repository name for ZoeDepth on torch.hub.
            zoe_model_name (str): Model name for ZoeDepth.
            fusion_out_channels (int): Number of output channels for fusion block.
            unet_bilinear (bool): Whether to use bilinear upsampling in UNet.
        """
        super(CombinedDepthModel, self).__init__()

        # Load ZoeDepth pre-trained model
        self.model_zoe = torch.hub.load(zoe_repo, zoe_model_name, pretrained=True)

        # Load DeepLabV3 model for semantic segmentation with ResNet-50 backbone
        segmentation_model = deeplabv3_resnet50(pretrained=True)
        self.segmentation_backbone = segmentation_model.backbone

        # Define channel dimensions
        seg_channels = 2048  # Output channels from ResNet-50 backbone
        zoe_channels = 1      # ZoeDepth outputs a single depth channel
        combined_channels = seg_channels + zoe_channels  # Total combined channels

        # Define fusion block
        self.fusion_block = SimpleFusionBlock(combined_channels, fusion_out_channels)

        # Define UNet decoder
        self.unet_decoder = UNetDecoder(in_channels=fusion_out_channels, out_channels=1, bilinear=unet_bilinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the depth map.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Predicted depth map of shape (B, 1, H, W).
        """
        # Extract depth features from ZoeDepth
        features_zoe = self._extract_depth_features(x)  # Expected shape: (B, 1, H, W)

        # Extract segmentation features from DeepLabV3 backbone
        features_seg = self._extract_segmentation_features(x)  # Shape: (B, 2048, H/8, W/8)

        # Resize segmentation features to match depth feature spatial dimensions
        features_seg_resized = self._resize_features(features_seg, features_zoe.shape[2:])  # (B, 2048, H, W)

        # Combine depth and segmentation features
        combined_features = torch.cat((features_zoe, features_seg_resized), dim=1)  # Shape: (B, 2049, H, W)

        # Fuse combined features
        fused_features = self.fusion_block(combined_features)  # Shape: (B, fusion_out_channels, H, W)

        # Pass fused features through UNet decoder to get depth map
        depth_map = self.unet_decoder(fused_features)  # Shape: (B, 1, H, W)

        return depth_map

    def _extract_depth_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts depth features using ZoeDepth model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Depth features tensor of shape (B, 1, H, W).
        """
        features_zoe = self.model_zoe.infer(x)  # Expected shape: (B, 1, H, W)

        if features_zoe.dim() == 3:
            features_zoe = features_zoe.unsqueeze(1)  # Ensure shape: (B, 1, H, W)
        elif features_zoe.dim() != 4 or features_zoe.size(1) != 1:
            raise ValueError(f"Unexpected shape for features_zoe: {features_zoe.shape}")

        return features_zoe

    def _extract_segmentation_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts segmentation features using DeepLabV3 backbone.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Segmentation features tensor of shape (B, 2048, H/8, W/8).
        """
        features_seg = self.segmentation_backbone(x)['out']  # Shape: (B, 2048, H/8, W/8)
        return features_seg

    def _resize_features(self, features: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Resizes features to match the target spatial dimensions.

        Args:
            features (torch.Tensor): Features tensor to be resized.
            target_size (tuple): Target spatial size (H, W).

        Returns:
            torch.Tensor: Resized features tensor.
        """
        if features.shape[2:] != target_size:
            features = F.interpolate(
                features,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        return features

    def unfreeze_backbone_layers(self, layers_to_unfreeze: List[str]) -> None:
        """
        Unfreeze specific layers of the backbone networks for fine-tuning.

        Args:
            layers_to_unfreeze (List[str]): List of layer names to unfreeze.
        """
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

    def freeze_backbone_layers(self, layers_to_freeze: List[str]) -> None:
        """
        Freeze specific layers of the backbone networks to prevent them from being updated during training.

        Args:
            layers_to_freeze (List[str]): List of layer names to freeze.
        """
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
