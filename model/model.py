import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.hub
from typing import List

from .unet_parts import DoubleConv, Down, Up, OutConv
from .unet_model import UNet

class AttentionFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionFusionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Cơ chế chú ý kênh
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 16)
        self.fc2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Áp dụng cơ chế chú ý kênh
        w = self.global_avg_pool(out)
        w = self.fc1(w)
        w = self.bn2(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)

        out = out * w

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.conv1 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x6 = self.atrous_block6(x)
        x12 = self.atrous_block12(x)
        x18 = self.atrous_block18(x)

        x = torch.cat([x1, x6, x12, x18], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)

class CombinedDepthModel(nn.Module):
    def __init__(
        self,
        zoe_repo: str = "isl-org/ZoeDepth",
        zoe_model_name: str = "ZoeD_N",
        fusion_out_channels: int = 256,
        unet_bilinear: bool = False
    ):
        super(CombinedDepthModel, self).__init__()

        # Load mô hình ZoeDepth pre-trained
        self.model_zoe = torch.hub.load(zoe_repo, zoe_model_name, pretrained=True)
        self.zoe_backbone = self.model_zoe.backbone  # Giả sử ZoeDepth có thuộc tính backbone

        # Load mô hình DeepLabV3 với ResNet-50 backbone
        segmentation_model = deeplabv3_resnet50(pretrained=True)
        self.segmentation_backbone = segmentation_model.backbone

        # Áp dụng ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        self.fusion_block = AttentionFusionBlock(
            in_channels=256 + 1024,  # Kết hợp kênh từ ZoeDepth và ASPP
            out_channels=fusion_out_channels
        )

        # UNet
        self.unet_decoder = UNet(
            n_channels=fusion_out_channels,
            n_classes=1,
            bilinear=unet_bilinear
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features_zoe = self._extract_depth_features(x)  # (B, 1024, H/4, W/4)

        features_seg = self._extract_segmentation_features(x)  # (B, 256, H/8, W/8)

        features_seg = F.interpolate(features_seg, size=features_zoe.shape[2:], mode='bilinear', align_corners=False)

        combined_features = torch.cat((features_zoe, features_seg), dim=1)

        fused_features = self.fusion_block(combined_features)

        depth_map = self.unet_decoder(fused_features)

        return depth_map

    def _extract_depth_features(self, x: torch.Tensor) -> torch.Tensor:
        # Trích xuất đặc trưng từ ZoeDepth backbone
        features = self.zoe_backbone(x)  # Cần đảm bảo ZoeDepth có phương thức này
        return features

    def _extract_segmentation_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.segmentation_backbone(x)['out'] 
        features = self.aspp(features)
        return features

    def unfreeze_backbone_layers(self, layers_to_unfreeze: List[str]) -> None:
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        for name, param in self.zoe_backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

    def freeze_backbone_layers(self, layers_to_freeze: List[str]) -> None:
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

        for name, param in self.zoe_backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
