""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 hoặc (convolution => ReLU) * 2 tùy thuộc vào `use_bn`"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_bn=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # mid_channels = in_channels // 2
            self.conv = DoubleConv(in_channels, out_channels, use_bn=use_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x2 is not None:
            # Xử lý kích thước nếu cần
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
