# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.hub
from typing import List

# Import các thành phần cần thiết từ DPT
from .dpt import _make_encoder, _make_fusion_block, forward_vit, Interpolate


class CombinedDepthModel(nn.Module):
    """
    Mô hình kết hợp ZoeDepth và DeepLabV3 với bộ giải mã DPT.
    """

    def __init__(
        self,
        zoe_repo: str = "isl-org/ZoeDepth",
        zoe_model_name: str = "ZoeD_N",
        backbone: str = "vitb_rn50_384",
        features: int = 256,
        readout: str = "project",
        use_bn: bool = True,
    ):
        super().__init__()

        # Tải mô hình ZoeDepth đã được huấn luyện trước
        self.model_zoe = torch.hub.load(zoe_repo, zoe_model_name, pretrained=True)

        # Tải backbone của DeepLabV3 cho phân đoạn ảnh
        segmentation_model = deeplabv3_resnet50(pretrained=True)
        self.segmentation_backbone = segmentation_model.backbone

        # Khởi tạo encoder và decoder của DPT
        self.pretrained, self.scratch = _make_encoder(
            backbone=backbone,
            features=features,
            use_pretrained=False,  # Sửa tham số thành 'use_pretrained' cho phù hợp với hàm _make_encoder
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

        # Định nghĩa output head như trong DPTDepthModel
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tiến hành truyền xuôi để tính toán bản đồ độ sâu.

        Args:
            x (torch.Tensor): Ảnh đầu vào dạng tensor với kích thước (B, 3, H, W).

        Returns:
            torch.Tensor: Bản đồ độ sâu dự đoán với kích thước (B, 1, H, W).
        """
        # Trích xuất đặc trưng độ sâu từ ZoeDepth
        features_zoe = self._extract_depth_features(x)  # (B, 1, H, W)

        # Trích xuất đặc trưng phân đoạn từ backbone của DeepLabV3
        features_seg = self._extract_segmentation_features(x)  # (B, 2048, H/8, W/8)

        # Thay đổi kích thước đặc trưng phân đoạn để khớp với đặc trưng độ sâu
        features_seg_resized = self._resize_features(features_seg, features_zoe.shape[2:])  # (B, 2048, H, W)

        # Kết hợp đặc trưng từ ZoeDepth và phân đoạn
        combined_features = torch.cat((features_zoe, features_seg_resized), dim=1)  # (B, C, H, W)

        # Điều chỉnh số kênh để phù hợp với input của encoder DPT (3 kênh)
        combined_features = self._adapt_channels(combined_features, target_channels=3)

        # Truyền qua encoder của DPT
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, combined_features)

        # Xử lý các layer qua decoder của DPT
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Tính toán bản đồ độ sâu đầu ra
        depth_map = self.scratch.output_conv(path_1)  # (B, 1, H, W)

        return depth_map

    def _adapt_channels(self, x: torch.Tensor, target_channels: int) -> torch.Tensor:
        """
        Điều chỉnh số kênh của tensor x thành target_channels.

        Args:
            x (torch.Tensor): Tensor đầu vào.
            target_channels (int): Số kênh mong muốn.

        Returns:
            torch.Tensor: Tensor với số kênh đã được điều chỉnh.
        """
        current_channels = x.shape[1]
        if current_channels > target_channels:
            x = x[:, :target_channels, :, :]
        elif current_channels < target_channels:
            pad = target_channels - current_channels
            padding = torch.zeros(
                x.shape[0], pad, x.shape[2], x.shape[3], device=x.device
            )
            x = torch.cat([x, padding], dim=1)
        return x

    def _extract_depth_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất đặc trưng độ sâu sử dụng mô hình ZoeDepth.

        Args:
            x (torch.Tensor): Ảnh đầu vào.

        Returns:
            torch.Tensor: Đặc trưng độ sâu.
        """
        features_zoe = self.model_zoe.infer(x)  # (B, 1, H, W)
        if features_zoe.dim() == 3:
            features_zoe = features_zoe.unsqueeze(1)
        return features_zoe

    def _extract_segmentation_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất đặc trưng phân đoạn sử dụng backbone của DeepLabV3.

        Args:
            x (torch.Tensor): Ảnh đầu vào.

        Returns:
            torch.Tensor: Đặc trưng phân đoạn.
        """
        features_seg = self.segmentation_backbone(x)['out']  # (B, 2048, H/8, W/8)
        return features_seg

    def _resize_features(self, features: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Thay đổi kích thước đặc trưng để khớp với kích thước không gian mục tiêu.

        Args:
            features (torch.Tensor): Đặc trưng cần thay đổi kích thước.
            target_size (tuple): Kích thước không gian mục tiêu (H, W).

        Returns:
            torch.Tensor: Đặc trưng đã được thay đổi kích thước.
        """
        return F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)

    def unfreeze_backbone_layers(self, layers_to_unfreeze: List[str]) -> None:
        """
        Mở khóa các lớp cụ thể của mạng backbone để fine-tune.

        Args:
            layers_to_unfreeze (List[str]): Danh sách tên các lớp cần mở khóa.
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
        Đóng băng các lớp cụ thể của mạng backbone để ngăn chúng được cập nhật trong quá trình huấn luyện.

        Args:
            layers_to_freeze (List[str]): Danh sách tên các lớp cần đóng băng.
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
