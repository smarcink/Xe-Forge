import math
from typing import Callable

import torch


def icnr(
    weight: torch.Tensor,
    bias: torch.Tensor,
    initializer: Callable | None = None,
    upscale_factor=2,
    *args,
    **kwargs,
):
    """Fill weight tensor using icnr (initialize conv as nearest resize) weight initialization
    (https://arxiv.org/abs/1707.02937)

    Args:
        weight (torch.Tensor): conv weight to fill with icnr init
        initializer (Callable): base weight initializer from torch.nn.init to use, if None pytorch conv default init is used
        upscale_factor (int, optional): _description_. Defaults to 2.
        *args, **kwargs: base initializer arguments
    """
    if initializer is None:
        # Default torch init https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L182
        initializer = torch.nn.init.kaiming_uniform_
        kwargs = {"a": math.sqrt(5)}

    upscale_factor_squared = upscale_factor * upscale_factor
    assert weight.shape[0] % upscale_factor_squared == 0, (
        "The size of the first dimension: "
        f"tensor.shape[0] = {weight.shape[0]}"
        " is not divisible by square of upscale_factor: "
        f"upscale_factor = {upscale_factor}"
    )
    sub_kernel = torch.empty(weight.shape[0] // upscale_factor_squared, *weight.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    weight.data.copy_(sub_kernel.repeat_interleave(upscale_factor_squared, dim=0))

    # setting bias to zero further reduce checkerboarding by pixelshuffle
    bias.data.copy_(torch.zeros_like(bias))
