import torch
import torch.nn as nn
from .unet_parts import Up, OutConv

class UNetHead(nn.Module):
    def __init__(self, n_classes, use_bn=True):
        super(UNetHead, self).__init__()
        self.up1 = Up(512, 256, use_bn=use_bn)  # 256 (x4 upsampled) + 256 (x3) = 512
        self.up2 = Up(512, 256, use_bn=use_bn)  # 256 (from up1) + 256 (x2) = 512
        self.up3 = Up(512, 256, use_bn=use_bn)  # 256 (from up2) + 256 (x1) = 512
        self.up4 = Up(256, 64, use_bn=use_bn)   # No skip connection
        self.outc_seg = OutConv(64, n_classes)
        self.outc_depth = OutConv(64, 1)

    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, None)  # Nếu không có skip connection, truyền None
        seg_output = self.outc_seg(x)
        depth_output = self.outc_depth(x)
        return seg_output, depth_output
