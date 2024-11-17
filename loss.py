# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss cho ước lượng độ sâu.
    """
    def __init__(self, variance_focus=0.85):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, input, target):
        """
        input: Bản đồ độ sâu dự đoán (B x 1 x H x W)
        target: Bản đồ độ sâu thực (B x 1 x H x W)
        """
        eps = 1e-8
        d = torch.log(input + eps) - torch.log(target + eps)
        mse = torch.mean(d ** 2)
        var = torch.var(d)
        return (self.variance_focus * var) + (1 - self.variance_focus) * mse


class L1Loss(nn.Module):
    """
    L1 Loss giữa bản đồ độ sâu dự đoán và thực.
    """
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, input, target):
        return self.l1_loss(input, target)


class SSIMLoss(nn.Module):
    """
    SSIM Loss cho ước lượng độ sâu.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, img1, img2):
        ssim_value = self._ssim(img1, img2)
        return 1 - ssim_value


class GradientLoss(nn.Module):
    """
    Gradient Loss để khuyến khích sự mượt mà trong bản đồ độ sâu.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        pred_d_dx, pred_d_dy = self.gradient(pred)
        target_d_dx, target_d_dy = self.gradient(target)

        loss_dx = torch.mean(torch.abs(pred_d_dx - target_d_dx))
        loss_dy = torch.mean(torch.abs(pred_d_dy - target_d_dy))

        return loss_dx + loss_dy

    def gradient(self, img):
        D_dx = img[:, :, :, :-1] - img[:, :, :, 1:]
        D_dy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return D_dx, D_dy


class DepthLoss(nn.Module):
    """
    Hàm loss tổng hợp cho ước lượng độ sâu.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super(DepthLoss, self).__init__()
        self.silog_loss = SILogLoss()
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        self.alpha = alpha  # Trọng số cho SiLog loss
        self.beta = beta    # Trọng số cho L1 loss
        self.gamma = gamma  # Trọng số cho SSIM loss
        self.delta = delta  # Trọng số cho Gradient loss

    def forward(self, pred, target):
        loss_silog = self.silog_loss(pred, target)
        loss_l1 = self.l1_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_gradient = self.gradient_loss(pred, target)

        total_loss = (self.alpha * loss_silog) + \
                     (self.beta * loss_l1) + \
                     (self.gamma * loss_ssim) + \
                     (self.delta * loss_gradient)

        return total_loss
