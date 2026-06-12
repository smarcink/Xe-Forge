from enum import Enum

import torch
import torch.nn as nn


class SigmoidFP32(nn.Module):
    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            return torch.sigmoid(x.float())


class ActivationType(Enum):
    NONE = "none"
    ELU = "elu"
    RELU = "relu"
    RELU6 = "relu6"
    CLIP = "clip"
    LEAKY_RELU = "leaky_relu"
    PARAMETRIC_RELU = "prelu"
    SIGMOID = "sigmoid"
    HARDSIGMOID = "hard_sigmoid"
    GELU = "gelu"

    def __str__(self):
        return self.value


def create_activation(atype: ActivationType, inplace: bool = False, **kwargs) -> nn.Module:
    if atype == ActivationType.NONE:
        return nn.Identity()
    elif atype == ActivationType.ELU:
        alpha = kwargs.get("elu_alpha", 1.0)
        return nn.ELU(alpha=alpha, inplace=inplace)
    elif atype == ActivationType.RELU:
        return nn.ReLU(inplace=inplace)
    elif atype == ActivationType.RELU6:
        return nn.ReLU6(inplace=inplace)
    elif atype == ActivationType.LEAKY_RELU:
        slope = kwargs.get("leaky_relu_slope", 0.02)
        return nn.LeakyReLU(negative_slope=slope, inplace=inplace)
    elif atype == ActivationType.PARAMETRIC_RELU:
        init_value = kwargs.get("prelu_init", 0.25)
        return nn.PReLU(init=init_value)
    elif atype == ActivationType.SIGMOID:
        return SigmoidFP32()
    elif atype == ActivationType.GELU:
        return nn.GELU()
    elif atype == ActivationType.HARDSIGMOID:
        return nn.Hardsigmoid()
    elif atype == ActivationType.CLIP:
        clip_min = kwargs.get("clip_min", 0.0)
        clip_max = kwargs.get("clip_max", 1.0)
        return nn.Hardtanh(min_val=clip_min, max_val=clip_max)
