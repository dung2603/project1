import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np

def denormalize(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        print("Params passed to Resize transform:")
        print("\twidth: ", width)
        print("\theight: ", height)
        print("\tresize_target: ", resize_target)
        print("\tkeep_aspect_ratio: ", keep_aspect_ratio)
        print("\tensure_multiple_of: ", ensure_multiple_of)
        print("\tresize_method: ", resize_method)

        self.__width = width
        self.__height = height

        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, x):
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (height, width), mode='bilinear', align_corners=True)

class PrepForZoeDepth(object):
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
    def __init__(self, zoe_model, trainable=False, fetch_features=False,
                 keep_aspect_ratio=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), 
                 img_size=384, depth_key='metric_depth', **kwargs):
        super().__init__()
        self.core = zoe_model
        self.trainable = trainable
        self.fetch_features = fetch_features
        self.layer_names = layer_names
        self.core_out = {}
        self.handles = [] 
        self.depth_key = depth_key
        self.set_trainable(trainable)
        self.prep = PrepForZoeDepth(keep_aspect_ratio=keep_aspect_ratio,
                                    img_size=img_size, do_resize=kwargs.get('do_resize', True))
        if kwargs.get('freeze_bn', False):
            self.freeze_bn()

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

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
        return self

    def forward(self, x, denorm=False, return_rel_depth=False):
        with torch.no_grad():
            if denorm:
                x = denormalize(x)
            x = self.prep(x)
            # print("Shape after prep: ", x.shape)

        with torch.set_grad_enabled(self.trainable):
            rel_depth = self.core(x)

            if isinstance(rel_depth, dict):
                rel_depth = rel_depth.get(self.depth_key, None)
                if rel_depth is None:
                    available_keys = list(rel_depth.keys()) if rel_depth else []
                    raise ValueError(f"ZoeCore.forward: '{self.depth_key}' key not found in rel_depth dict. Available keys: {available_keys}")
            
            if not self.fetch_features:
                return rel_depth
            out = [self.core_out[k] for k in self.layer_names]

            if return_rel_depth:
                return rel_depth, out
            return out

    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, zoe_model):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(list(zoe_model.scratch.output_conv.children())[3].register_forward_hook(get_activation("out_conv", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(zoe_model.scratch.refinenet4.register_forward_hook(get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(zoe_model.scratch.refinenet3.register_forward_hook(get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(zoe_model.scratch.refinenet2.register_forward_hook(get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(zoe_model.scratch.refinenet1.register_forward_hook(get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(zoe_model.scratch.layer4_rn.register_forward_hook(get_activation("l4_rn", self.core_out)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = [] 
        return self

    def __del__(self):
        self.remove_hooks()

    @staticmethod
    def build(zoe_model_name="ZoeD_N", trainable=False, use_pretrained=True, fetch_features=False, freeze_bn=True, keep_aspect_ratio=True, img_size=384, depth_key='metric_depth', **kwargs):
        # Load ZoeDepth model
        zoe_model = torch.hub.load("isl-org/ZoeDepth", zoe_model_name, pretrained=use_pretrained)
        print(zoe_model)
        zoe_core = ZoeCore(zoe_model, trainable=trainable, fetch_features=fetch_features,
                           freeze_bn=freeze_bn, keep_aspect_ratio=keep_aspect_ratio, 
                           img_size=img_size, depth_key=depth_key, **kwargs)
        return zoe_core
