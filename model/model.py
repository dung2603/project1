# model.py

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from typing import List

from .unet_model import UNet
from .zoedepthcore import ZoeCore

class SimpleFusionBlock(nn.Module)
    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleFusionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNetDecoder, self).__init__()
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=bilinear)

    def forward(self, x):
        return self.unet(x)

class CombinedDepthModel(nn.Module):
    def __init__(self, core, semantic_feature_channels: int, unet_out_channels: int = 1, bilinear: bool = False):
        super(CombinedDepthModel, self).__init__()
        self.core = core

        # Initialize semantic segmentation backbone using DeepLabV3 ResNet50 with pretrained weights
        self.segmentation_backbone = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.segmentation_backbone.eval()  # Đặt mô hình ở chế độ đánh giá
        for param in self.segmentation_backbone.parameters():
            param.requires_grad = False  # Đóng băng các tham số

        if not hasattr(core, 'output_channels'):
            raise AttributeError("ZoeCore object has no attribute 'output_channels'. Hãy đảm bảo rằng ZoeCore có thuộc tính này.")

        C_zoe = core.output_channels[0] 
        self.fusion_block = SimpleFusionBlock(in_channels=C_zoe + semantic_feature_channels, out_channels=128)

        # Khởi tạo UNet Decoder
        self.unet_decoder = UNetDecoder(in_channels=128, out_channels=unet_out_channels, bilinear=bilinear)

    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        _, _, h, w = x.shape
        self.orig_input_width = w
        self.orig_input_height = h

      
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)
        outconv_activation = out[0]
        last = outconv_activation  # Shape: (B, C_zoe, H', W')

        with torch.no_grad():
            segmentation_output = self.segmentation_backbone(x)['out']  # Shape: (B, S, H', W')

        semantic_features = segmentation_output
        semantic_features = self._resize_features(semantic_features, last.shape[2:])

        combined_features = torch.cat([last, semantic_features], dim=1)  # Shape: (B, C_zoe + S, H', W')
        fused_features = self.fusion_block(combined_features)  # Shape: (B, 128, H', W')

     
        metric_depth = self.unet_decoder(fused_features)  # Shape: (B, 1, H, W)

        output = dict(metric_depth=metric_depth)

        if return_final_centers or return_probs:
            # Đảm bảo ZoeCore trả về 'bin_centers' trong 'out'
            if len(out) > 1:
                bin_centers = out[1]
                bin_centers = self._resize_features(bin_centers, metric_depth.shape[2:])
                output['bin_centers'] = bin_centers

        if return_probs:
            if len(out) > 2:
                probs = out[2]
                probs = self._resize_features(probs, metric_depth.shape[2:])
                output['probs'] = probs

        return output

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

        for name, param in self.core.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

    def freeze_backbone_layers(self, layers_to_freeze: List[str]) -> None:
        for name, param in self.segmentation_backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

        for name, param in self.core.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

    @staticmethod
    def build(zoe_model_name="ZoeD_N", trainable=False, use_pretrained=True, fetch_features=False, freeze_bn=True, 
              keep_aspect_ratio=True, img_size=384, semantic_feature_channels: int = 21, unet_out_channels: int = 1, bilinear: bool = False, **kwargs):
        zoe_model = torch.hub.load(
            "isl-org/ZoeDepth",
            zoe_model_name,
            pretrained=use_pretrained,
            trust_repo=True
        )
        print(zoe_model)
        
        # Create ZoeCore
        core = ZoeCore(
            zoe_model,
            trainable=trainable,
            fetch_features=fetch_features,
            freeze_bn=freeze_bn,
            keep_aspect_ratio=keep_aspect_ratio, 
            img_size=img_size,
            **kwargs
        )
        
        combined_model = CombinedDepthModel(
            core=core,
            semantic_feature_channels=semantic_feature_channels, 
            unet_out_channels=unet_out_channels,
            bilinear=bilinear,
            **kwargs
        )
        return combined_model
