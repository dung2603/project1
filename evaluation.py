import torch
import numpy as np

def mask_depth( target, min_depth=1e-3, max_depth=10):
    """
    Tạo mặt nạ để loại bỏ các giá trị không hợp lệ trong bản đồ độ sâu.

    :param pred: Bản đồ độ sâu dự đoán (B, 1, H, W)
    :param target: Bản đồ độ sâu thực tế (B, 1, H, W)
    :param min_depth: Giá trị độ sâu tối thiểu để tính toán
    :param max_depth: Giá trị độ sâu tối đa để tính toán
    :return: Mặt nạ boolean với kích thước (B, H, W)
    """
    mask = (target > min_depth) & (target < max_depth)
    return mask

def compute_mae(pred, target, mask=None):
    """
    Tính toán Mean Absolute Error (MAE).

    :param pred: Bản đồ độ sâu dự đoán (B, 1, H, W)
    :param target: Bản đồ độ sâu thực tế (B, 1, H, W)
    :param mask: Mặt nạ boolean để loại bỏ các giá trị không hợp lệ (B, H, W)
    :return: Giá trị MAE trung bình
    """
    pred = pred.squeeze()
    target = target.squeeze()

    if mask is None:
        mask = target > 0

    mae = torch.abs(pred - target)[mask].mean().item()
    return mae

def compute_rmse(pred, target, mask=None):
    """
    Tính toán Root Mean Squared Error (RMSE).

    :param pred: Bản đồ độ sâu dự đoán (B, 1, H, W)
    :param target: Bản đồ độ sâu thực tế (B, 1, H, W)
    :param mask: Mặt nạ boolean để loại bỏ các giá trị không hợp lệ (B, H, W)
    :return: Giá trị RMSE trung bình
    """
    pred = pred.squeeze()
    target = target.squeeze()

    if mask is None:
        mask = target > 0

    rmse = torch.sqrt(torch.mean((pred - target) ** 2)[mask]).item()
    return rmse

def compute_abs_rel(pred, target, mask=None):
    """
    Tính toán Absolute Relative Error (Abs Rel).

    :param pred: Bản đồ độ sâu dự đoán (B, 1, H, W)
    :param target: Bản đồ độ sâu thực tế (B, 1, H, W)
    :param mask: Mặt nạ boolean để loại bỏ các giá trị không hợp lệ (B, H, W)
    :return: Giá trị Abs Rel trung bình
    """
    pred = pred.squeeze()
    target = target.squeeze()

    if mask is None:
        mask = target > 0

    abs_rel = torch.mean(torch.abs(pred - target) / target)[mask].item()
    return abs_rel

def compute_delta(pred, target, mask=None, delta=1.25):
    """
    Tính toán tỷ lệ Accuracy với các ngưỡng δ < 1.25, δ < 1.25², δ < 1.25³.

    :param pred: Bản đồ độ sâu dự đoán (B, 1, H, W)
    :param target: Bản đồ độ sâu thực tế (B, 1, H, W)
    :param mask: Mặt nạ boolean để loại bỏ các giá trị không hợp lệ (B, H, W)
    :param delta: Ngưỡng delta (ví dụ: 1.25, 1.25², 1.25³)
    :return: Giá trị tỷ lệ accuracy
    """
    pred = pred.squeeze()
    target = target.squeeze()

    if mask is None:
        mask = target > 0

    ratio = torch.max(pred / target, target / pred)
    delta_accuracy = torch.mean((ratio < delta).float())[mask].item()
    return delta_accuracy

def compute_all_metrics(pred, target, mask=None, delta_values=[1.25, 1.25**2, 1.25**3]):
    """
    Tính toán tất cả các chỉ số lỗi cho bản đồ độ sâu dự đoán và thực tế.

    :param pred: Bản đồ độ sâu dự đoán (B, 1, H, W)
    :param target: Bản đồ độ sâu thực tế (B, 1, H, W)
    :param mask: Mặt nạ boolean để loại bỏ các giá trị không hợp lệ (B, H, W)
    :param delta_values: Danh sách các giá trị delta để tính accuracy
    :return: Dictionary chứa các chỉ số lỗi
    """
    mae = compute_mae(pred, target, mask)
    rmse = compute_rmse(pred, target, mask)
    abs_rel = compute_abs_rel(pred, target, mask)

    delta_metrics = {}
    for delta in delta_values:
        delta_key = f"delta<{delta}"
        delta_metrics[delta_key] = compute_delta(pred, target, mask, delta)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'abs_rel': abs_rel,
    }
    metrics.update(delta_metrics)

    return metrics

if __name__ == "__main__":
    # Ví dụ sử dụng các hàm trong metrics.py
    import torch

    # Tạo các tensor giả lập
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    target = torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]])  # Shape: (1, 1, 2, 2)

    # Tạo mặt nạ
    mask = mask_depth(pred, target)

    # Tính các chỉ số
    mae = compute_mae(pred, target, mask)
    rmse = compute_rmse(pred, target, mask)
    abs_rel = compute_abs_rel(pred, target, mask)
    delta1 = compute_delta(pred, target, mask, delta=1.25)
    delta2 = compute_delta(pred, target, mask, delta=1.25**2)
    delta3 = compute_delta(pred, target, mask, delta=1.25**3)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"Abs Rel: {abs_rel}")
    print(f"Delta < 1.25: {delta1}")
    print(f"Delta < 1.25^2: {delta2}")
    print(f"Delta < 1.25^3: {delta3}")

    # Tính tất cả các chỉ số cùng một lúc
    all_metrics = compute_all_metrics(pred, target, mask)
    print("All Metrics:", all_metrics)