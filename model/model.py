# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.hub
from typing import List

# Import các thành phần từ DPT
from .dpt import DPTDepthModel, _make_fusion_block, _make_encoder, forward_vit, Interpolate


class CombinedDepthModel(nn.Module):
    """
    Mô hình kết hợp ZoeDepth và DeepLabV3 với bộ giải mã DPT.
    """

    def __init__(
        self,
        zoe_repo: str = "isl-org/ZoeDepth",
        zoe_model_name: str = "ZoeD_N",
        backbone: str = "vitb_rn50_384",  # Sử dụng backbone của DPT
        features: int = 256,
        readout: str = "project",
        use_bn: bool = True,
        fusion_out_channels: int = 256,  # Điều chỉnh để phù hợp với DPT
    ):
        super(CombinedDepthModel, self).__init__()

        # Load ZoeDepth pre-trained model
        self.model_zoe = torch.hub.load(zoe_repo, zoe_model_name, pretrained=True)

        # Load DeepLabV3 model for semantic segmentation with ResNet-50 backbone
        segmentation_model = deeplabv3_resnet50(pretrained=True)
        self.segmentation_backbone = segmentation_model.backbone

        # DPT encoder
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            pretrained=False,
            groups=1,
            expand=False,
            exportable=False,
            hooks=None,
            use_readout=readout,
            enable_attention_hooks=False,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Define head as in DPTDepthModel
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.scratch.output_conv = self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract depth features from ZoeDepth
        features_zoe = self._extract_depth_features(x)  # Shape: (B, 1, H, W)

        # Extract segmentation features from DeepLabV3 backbone
        features_seg = self._extract_segmentation_features(x)  # Shape: (B, 2048, H/8, W/8)

        # Resize features to match
        features_seg_resized = self._resize_features(features_seg, features_zoe.shape[2:])

        # Combine features
        combined_features = torch.cat((features_zoe, features_seg_resized), dim=1)  # Shape: (B, C, H, W)

        # Pass through DPT encoder
        if combined_features.shape[1] != 3:
            # Nếu số kênh không phải là 3, chuyển đổi về 3 kênh
            combined_features = self._adapt_channels(combined_features, 3)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, combined_features)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        depth_map = self.scratch.output_conv(path_1)  # Shape: (B, 1, H, W)

        return depth_map

    def _adapt_channels(self, x: torch.Tensor, target_channels: int) -> torch.Tensor:
        """
        Chuyển đổi số kênh của tensor x về target_channels.
        """
        if x.shape[1] > target_channels:
            x = x[:, :target_channels, :, :]
        elif x.shape[1] < target_channels:
            pad = target_channels - x.shape[1]
            x = torch.cat([x, torch.zeros(x.shape[0], pad, x.shape[2], x.shape[3], device=x.device)], dim=1)
        return x

    def _extract_depth_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts depth features using ZoeDepth model.
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
        """
        features_seg = self.segmentation_backbone(x)['out']  # Shape: (B, 2048, H/8, W/8)
        return features_seg

    def _resize_features(self, features: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Resizes features to match the target spatial dimensions.
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
        """
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        for name, param in self.pretrained.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

    def freeze_backbone_layers(self, layers_to_freeze: List[str]) -> None:
        """
        Freeze specific layers of the backbone networks to prevent them from being updated during training.
        """
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

        for name, param in self.model_zoe.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

        for name, param in self.pretrained.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
