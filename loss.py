# loss.py

import torch
import torch.nn as nn

class SILogLoss(nn.Module):
    def __init__(self, variance_focus=0.85, eps=1e-8):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.eps = eps

    def forward(self, input, target):
        """
        input: Bản đồ độ sâu dự đoán (B x 1 x H x W)
        target: Bản đồ độ sâu thực (B x 1 x H x W)
        """
        # Thêm epsilon để tránh log(0)
        d = torch.log(input + self.eps) - torch.log(target + self.eps)
        mse = torch.mean(d ** 2)
        var = torch.var(d)
        return (self.variance_focus * var) + (1 - self.variance_focus) * mse


class GradientLoss(nn.Module):
   
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        pred_d_dx, pred_d_dy = self.gradient(pred)
        target_d_dx, target_d_dy = self.gradient(target)

        loss_dx = torch.mean(torch.abs(pred_d_dx - target_d_dx))
        loss_dy = torch.mean(torch.abs(pred_d_dy - target_d_dy))

        return loss_dx + loss_dy

    def gradient(self, img):
        """
        Tính gradient theo trục x và y.
        """
        D_dx = img[:, :, :, :-1] - img[:, :, :, 1:]
        D_dy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return D_dx, D_dy


class DepthLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, delta=0.1):
        super(DepthLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.silog_loss = SILogLoss()
        self.gradient_loss = GradientLoss()
        self.alpha = alpha  # Trọng số cho MSELoss
        self.beta = beta    # Trọng số cho SILogLoss
        self.delta = delta  # Trọng số cho GradientLoss

    def forward(self, pred, target):
        loss_mse = self.mse_loss(pred, target)
        loss_silog = self.silog_loss(pred, target)
        loss_grad = self.gradient_loss(pred, target)

        total_loss = (self.alpha * loss_mse) + (self.beta * loss_silog) + (self.delta * loss_grad)
        return total_loss

