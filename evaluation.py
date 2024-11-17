# metric.py

import torch

def compute_metrics(pred, gt, mask=None):
    """
    Compute depth estimation metrics between predicted and ground truth depths.

    Args:
        pred (torch.Tensor): Predicted depth map of shape (B, 1, H, W)
        gt (torch.Tensor): Ground truth depth map of shape (B, 1, H, W)
        mask (torch.Tensor): Optional mask of shape (B, 1, H, W) where True indicates valid pixels

    Returns:
        dict: Dictionary containing computed metrics
    """
    # Ensure pred and gt have the same shape
    assert pred.shape == gt.shape, "Predicted and ground truth depths must have the same shape"

    # Flatten tensors and apply mask if provided
    if mask is not None:
        valid_mask = mask.bool()
        pred = pred[valid_mask]
        gt = gt[valid_mask]
    else:
        pred = pred.view(-1)
        gt = gt.view(-1)

    # Avoid division by zero and log of zero
    epsilon = 1e-6
    pred = torch.clamp(pred, min=epsilon)
    gt = torch.clamp(gt, min=epsilon)

    # Compute absolute relative error
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    # Compute RMSE
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))

    # Compute log10 error
    log10 = torch.mean(torch.abs(torch.log10(gt) - torch.log10(pred)))

    # Compute threshold accuracy
    max_ratio = torch.max(pred / gt, gt / pred)
    delta = max_ratio
    a1 = (delta < 1.25).float().mean()
    a2 = (delta < 1.25 ** 2).float().mean()
    a3 = (delta < 1.25 ** 3).float().mean()

    metrics = {
        'abs_rel': abs_rel.item(),
        'rmse': rmse.item(),
        'log10': log10.item(),
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item(),
    }

    return metrics
