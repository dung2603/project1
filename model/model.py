# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from typing import List
from .unet_model import UNet
# Import ZoeCore
from .zoedepthcore import ZoeCore  # Điều chỉnh đường dẫn import nếu cần

class SimpleFusionBlock(nn.Module):
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
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(UNetDecoder, self).__init__()
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=bilinear)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

class CombinedDepthModel(nn.Module):
    def __init__(
        self,
        zoe_model_name: str = "ZoeD_N",
        fusion_out_channels: int = 128,
        unet_bilinear: bool = False,
        depth_key: str = "metric_depth" 
    ):
        super(CombinedDepthModel, self).__init__()
        self.model_zoe = ZoeCore.build(
            zoe_model_name=zoe_model_name,
            trainable=False,
            use_pretrained=True,
            fetch_features=False, 
            freeze_bn=True,
            keep_aspect_ratio=True,
            img_size=384,
            depth_key=depth_key 
        )
        segmentation_model = deeplabv3_resnet50(pretrained=True)
        self.segmentation_backbone = segmentation_model.backbone

        self.segmentation_backbone.eval()
        for param in self.segmentation_backbone.parameters():
            param.requires_grad = False

        seg_channels = 2048 
        zoe_channels = 1      
        combined_channels = seg_channels + zoe_channels 

        # Định nghĩa fusion block
        self.fusion_block = SimpleFusionBlock(combined_channels, fusion_out_channels)

        # Định nghĩa UNet decoder
        self.unet_decoder = UNetDecoder(in_channels=fusion_out_channels, out_channels=1, bilinear=unet_bilinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model_zoe.eval()
        self.segmentation_backbone.eval()

        with torch.no_grad():
            features_zoe = self._extract_depth_features(x) 

            features_seg = self._extract_segmentation_features(x)  # Shape: (B, 2048, H/16, W/16)

        features_seg_resized = self._resize_features(features_seg, features_zoe.shape[2:])  # (B, 2048, H, W)
        combined_features = torch.cat((features_zoe, features_seg_resized), dim=1)  # Shape: (B, 2049, H, W)
        fused_features = self.fusion_block(combined_features)  # Shape: (B, fusion_out_channels, H, W)
        depth_map = self.unet_decoder(fused_features)  # Shape: (B, 1, H, W)

        return depth_map

    def _extract_depth_features(self, x: torch.Tensor) -> torch.Tensor:
        features_zoe = self.model_zoe(x)  # ZoeCore's forward method

        if not isinstance(features_zoe, torch.Tensor):
            raise ValueError(f"Unexpected type for features_zoe: {type(features_zoe)}. Expected torch.Tensor.")

        if features_zoe.dim() == 3:
            features_zoe = features_zoe.unsqueeze(1)  # Ensure shape: (B, 1, H, W)
        elif features_zoe.dim() != 4 or features_zoe.size(1) != 1:
            raise ValueError(f"Unexpected shape for features_zoe: {features_zoe.shape}")

        return features_zoe

    def _extract_segmentation_features(self, x: torch.Tensor) -> torch.Tensor:
        features_seg_dict = self.segmentation_backbone(x)

        # Sử dụng khóa 'out' thay vì 'layer4'
        features_seg = features_seg_dict['out']  # Điều chỉnh khóa này nếu cần thiết
        return features_seg

    def _resize_features(self, features: torch.Tensor, target_size: tuple) -> torch.Tensor:
        if features.shape[2:] != target_size:
            features = F.interpolate(
                features,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        return features

    def unfreeze_backbone_layers(self, layers_to_unfreeze: List[str]) -> None:
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

    def freeze_backbone_layers(self, layers_to_freeze: List[str]) -> None
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
