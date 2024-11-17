# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DepthLoss(nn.Module):
    """
    Tổng hợp các hàm mất mát cho bài toán dự đoán độ sâu.
    Bao gồm L1 Loss và L2 Loss. Trọng số alpha xác định tỉ lệ kết hợp giữa hai hàm mất mát này.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Khởi tạo DepthLoss.

        Args:
            alpha (float): Trọng số cân bằng giữa L1 và L2 Loss.
                           alpha = 1.0 sử dụng chỉ L1 Loss,
                           alpha = 0.0 sử dụng chỉ L2 Loss,
                           0 < alpha < 1.0 sử dụng kết hợp cả hai.
        """
        super(DepthLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Tính toán tổng hợp L1 và L2 Loss giữa dự đoán và mục tiêu.

        Args:
            pred (torch.Tensor): Bản đồ độ sâu dự đoán (B, 1, H, W).
            target (torch.Tensor): Bản đồ độ sâu thực tế (B, 1, H, W).

        Returns:
            torch.Tensor: Giá trị mất mát tổng hợp.
        """
        if pred.size() != target.size():
            raise ValueError(f"Kích thước của pred {pred.size()} và target {target.size()} không khớp.")

        loss_l1 = self.l1_loss(pred, target)
        loss_l2 = self.l2_loss(pred, target)
        loss = self.alpha * loss_l1 + (1 - self.alpha) * loss_l2
        return loss


class SSIMLoss(nn.Module):
    """
    Hàm mất mát SSIM (Structural Similarity Index Measure).
    SSIM đánh giá sự giống nhau giữa hai hình ảnh dựa trên cấu trúc, độ sáng và độ tương phản.
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        """
        Khởi tạo SSIMLoss.

        Args:
            window_size (int): Kích thước cửa sổ Gaussian.
            size_average (bool): Nếu True, trung bình các giá trị SSIM trên toàn bộ ảnh.
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    @staticmethod
    def gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """
        Tạo một cửa sổ Gaussian 1D.

        Args:
            window_size (int): Kích thước cửa sổ.
            sigma (float): Độ lệch chuẩn của Gaussian.

        Returns:
            torch.Tensor: Cửa sổ Gaussian 1D.
        """
        gauss = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """
        Tạo một cửa sổ Gaussian 2D và mở rộng nó cho tất cả các kênh.

        Args:
            window_size (int): Kích thước cửa sổ.
            channel (int): Số lượng kênh.

        Returns:
            torch.Tensor: Cửa sổ Gaussian 2D mở rộng cho tất cả các kênh.
        """
        _1D_window = self.gaussian_window(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Tính toán mất mát SSIM giữa hai ảnh.

        Args:
            img1 (torch.Tensor): Ảnh đầu tiên (B, C, H, W).
            img2 (torch.Tensor): Ảnh thứ hai (B, C, H, W).

        Returns:
            torch.Tensor: Giá trị mất mát SSIM.
        """
        (_, channel, _, _) = img1.size()
        if channel != self.channel or self.window.data.type() != img1.data.type():
            self.window = self.create_window(self.window_size, channel).to(img1.device)
            self.channel = channel

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class SmoothnessLoss(nn.Module):
    """
    Hàm mất mát Smoothness để đảm bảo rằng bản đồ độ sâu mượt mà.
    """

    def __init__(self):
        """
        Khởi tạo SmoothnessLoss.
        """
        super(SmoothnessLoss, self).__init__()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Tính toán mất mát smoothness dựa trên đạo hàm của bản đồ độ sâu.

        Args:
            pred (torch.Tensor): Bản đồ độ sâu dự đoán (B, 1, H, W).

        Returns:
            torch.Tensor: Giá trị mất mát smoothness.
        """
        dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        loss = dx.mean() + dy.mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Kết hợp nhiều hàm mất mát cho bài toán dự đoán độ sâu.
    Bao gồm DepthLoss, SSIMLoss và SmoothnessLoss.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.1):
        """
        Khởi tạo CombinedLoss.

        Args:
            alpha (float): Trọng số cho DepthLoss.
            beta (float): Trọng số cho SSIMLoss.
        """
        super(CombinedLoss, self).__init__()
        self.depth_loss = DepthLoss(alpha=alpha)
        self.ssim_loss = SSIMLoss()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Tính toán tổng hợp các hàm mất mát.

        Args:
            pred (torch.Tensor): Bản đồ độ sâu dự đoán (B, 1, H, W).
            target (torch.Tensor): Bản đồ độ sâu thực tế (B, 1, H, W).

        Returns:
            torch.Tensor: Giá trị mất mát tổng hợp.
        """
        loss_depth = self.depth_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss = loss_depth + self.beta * loss_ssim
        return loss

