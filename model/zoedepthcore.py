import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

def denormalize(x):
    """Đảo ngược quá trình chuẩn hóa ImageNet áp dụng cho đầu vào.

    Args:
        x (torch.Tensor): Tensor đầu vào có dạng (N, 3, H, W)

    Returns:
        torch.Tensor: Tensor đã được đảo ngược chuẩn hóa
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean

class Resize(object):
    """Thay đổi kích thước của ảnh về kích thước mong muốn."""
    def __init__(self, width, height, resize_target=True, keep_aspect_ratio=False,
                 ensure_multiple_of=1, resize_method="lower_bound"):
        self.width = width
        self.height = height
        self.resize_target = resize_target
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (round(x / self.ensure_multiple_of) * self.ensure_multiple_of)
        if max_val is not None and y > max_val:
            y = (x // self.ensure_multiple_of) * self.ensure_multiple_of
        if y < min_val:
            y = (x // self.ensure_multiple_of + 1) * self.ensure_multiple_of
        return y

    def get_size(self, width, height):
        scale_height = self.height / height
        scale_width = self.width / width

        if self.keep_aspect_ratio:
            if self.resize_method == "lower_bound":
                scale = max(scale_width, scale_height)
            elif self.resize_method == "upper_bound":
                scale = min(scale_width, scale_height)
            else:
                scale = min(abs(1 - scale_width), abs(1 - scale_height))
            new_height = self.constrain_to_multiple_of(scale * height, min_val=self.height)
            new_width = self.constrain_to_multiple_of(scale * width, min_val=self.width)
        else:
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        return int(new_width), int(new_height)

    def __call__(self, x):
        width, height = x.shape[-1], x.shape[-2]
        new_width, new_height = self.get_size(width, height)
        return F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True)

class PrepForZoeDepth(object):
    """Chuẩn bị dữ liệu đầu vào cho ZoeDepth."""
    def __init__(self, resize_mode="minimal", keep_aspect_ratio=True, img_size=384, do_resize=True):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        self.normalization = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resizer = Resize(net_w, net_h, keep_aspect_ratio=keep_aspect_ratio,
                              ensure_multiple_of=32, resize_method=resize_mode) if do_resize else nn.Identity()

    def __call__(self, x):
        x = self.resizer(x)
        x = self.normalization(x)
        return x

class ZoeCore(nn.Module):
    """Lớp ZoeCore dùng để trích xuất đặc trưng từ ZoeDepth."""
    def __init__(self, zoe_model, trainable=False, fetch_features=False,
                 keep_aspect_ratio=True, img_size=384, **kwargs):
        super().__init__()
        self.core = zoe_model
        self.trainable = trainable
        self.fetch_features = fetch_features
        self.set_trainable(trainable)
        self.prep = PrepForZoeDepth(keep_aspect_ratio=keep_aspect_ratio,
                                    img_size=img_size, do_resize=kwargs.get('do_resize', True))
        if kwargs.get('freeze_bn', False):
            self.freeze_bn()

    def set_trainable(self, trainable):
        self.trainable = trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, denorm=False):
        if denorm:
            x = denormalize(x)
        x = self.prep(x)
        with torch.set_grad_enabled(self.trainable):
            rel_depth = self.core.infer(x)
        return rel_depth

    @staticmethod
    def build(zoe_model_name="ZoeD_N", trainable=False, use_pretrained=True, fetch_features=False, freeze_bn=True, keep_aspect_ratio=True, img_size=384, **kwargs):
        # Load ZoeDepth model
        zoe_model = torch.hub.load("isl-org/ZoeDepth", zoe_model_name, pretrained=use_pretrained)
        zoe_core = ZoeCore(zoe_model, trainable=trainable, fetch_features=fetch_features,
                           freeze_bn=freeze_bn, keep_aspect_ratio=keep_aspect_ratio, img_size=img_size, **kwargs)
        return zoe_core
