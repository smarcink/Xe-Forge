# Original: Model

import enum
import math
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ============================================================================
# Inlined dependencies (originally activations.py / initializer.py /
# quantization.py) so this file compiles standalone with no sibling imports.
# Only the pieces ConvBlock / ModelNew actually use are kept.
# ============================================================================

DEFAULT_QUANT_THRESHOLD = 3.0


# --- activations.py ---------------------------------------------------------
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


# --- initializer.py ---------------------------------------------------------
def icnr(
    weight: torch.Tensor,
    bias: torch.Tensor,
    initializer: Callable | None = None,
    upscale_factor=2,
    *args,
    **kwargs,
):
    """Fill weight tensor using icnr (initialize conv as nearest resize) init."""
    if initializer is None:
        initializer = torch.nn.init.kaiming_uniform_
        kwargs = {"a": math.sqrt(5)}

    upscale_factor_squared = upscale_factor * upscale_factor
    assert weight.shape[0] % upscale_factor_squared == 0
    sub_kernel = torch.empty(weight.shape[0] // upscale_factor_squared, *weight.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    weight.data.copy_(sub_kernel.repeat_interleave(upscale_factor_squared, dim=0))
    bias.data.copy_(torch.zeros_like(bias))


# --- quantization.py --------------------------------------------------------
class QFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class QRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class QTrunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.trunc(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class QuantUnsigned(nn.Module):
    def __init__(self, num_bits: int, qround: Type[torch.autograd.Function] = QTrunc):
        super().__init__()
        self.qround = qround.apply
        levels = 2**num_bits
        self._scale = 0.00392157
        self.qmax = levels - 1

    def _clamp(self, x):
        return torch.clamp(x, min=0, max=self.qmax)

    def _quantize(self, x, step):
        return self.integer(x * 255)

    def integer(self, x):
        return self._clamp(self.qround(x))

    def get_quant_step(self, max):
        return max * self._scale

    def get_quant_val(self, x, max):
        step = self.get_quant_step(max)
        return self._quantize(x, step)

    def forward(self, x, max):
        step = self.get_quant_step(max.float())
        xq = self._quantize(x.float(), step)
        return (xq * step).to(x.dtype)


class QuantSymmetric(nn.Module):
    def __init__(self, num_bits: int, qround: Type[torch.autograd.Function] = QTrunc):
        super().__init__()
        self.qround = qround.apply
        levels = 2**num_bits
        self.scale = 1 / (levels - 1)
        self.qmin = -levels // 2
        self.qmax = levels // 2 - 1

    def _clamp(self, x):
        return torch.clamp(x, min=self.qmin, max=self.qmax)

    def _quantize(self, x, step):
        return self.integer(x / step)

    def integer(self, x):
        return self._clamp(self.qround(x))

    def get_quant_step(self, max):
        return 2 * max * self.scale

    def get_quant_val(self, x, max):
        step = self.get_quant_step(max)
        return self._quantize(x, step)

    def forward(self, x, max):
        step = self.get_quant_step(max.float())
        xq = self._quantize(x.float(), step)
        return (xq * step).to(x.dtype)


class FixedQuant(nn.Module):
    def __init__(self, quantize=False, quantizer=None, max_val=DEFAULT_QUANT_THRESHOLD):
        super().__init__()
        self.quantize = quantize
        self.quantizer = quantizer
        self.max_val = max_val
        if self.quantize:
            assert self.quantizer is not None

    def get_quant_step(self):
        return self.quantizer.get_quant_step(self.max_val)

    def integer(self, x):
        return self.quantizer.get_quant_val(x, max=self.max_val)

    def forward(self, x):
        if self.quantize:
            return self.quantizer(x, max=self.max_val)
        return x


class TrainedQuant(nn.Module):
    def __init__(self, quantize=False, quantizer=None, max_val=DEFAULT_QUANT_THRESHOLD):
        super().__init__()
        self.quantize = quantize
        self.quantizer = quantizer
        max_log = torch.log(torch.abs(torch.tensor(max_val)))
        self.max_log = nn.Parameter(max_log, requires_grad=quantize)
        if self.quantize:
            assert self.quantizer is not None

    def _get_threshold(self):
        return torch.exp(self.max_log)

    def get_quant_step(self):
        if not self.quantize:
            raise Exception("Quantization is not enabled")
        return self.quantizer.get_quant_step(self._get_threshold())

    def integer(self, x):
        return self.quantizer.integer(x)

    def forward(self, x):
        if self.quantize:
            return self.quantizer(x, max=self._get_threshold())
        return x


class QConv2D(nn.Conv2d):
    def __init__(self, *args, quantize=False, num_bits=8, round_fn=QTrunc, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = quantize
        self.quantizer = QuantSymmetric(num_bits, round_fn)

    def _get_lim_per_channel(self):
        wt_size = self.weight.size()
        weight_flat = self.weight.view(wt_size[0], wt_size[1] * wt_size[2] * wt_size[3])
        (wt_min, _) = torch.min(weight_flat, dim=1)
        (wt_max, _) = torch.max(weight_flat, dim=1)
        wt_abs = torch.max(torch.abs(wt_min), torch.abs(wt_max))
        return wt_abs.view(wt_size[0], 1, 1, 1)

    def get_quant_step(self):
        return self.quantizer.get_quant_step(self._get_lim_per_channel())

    def get_quant_weights(self):
        if self.quantize:
            wt_max = self._get_lim_per_channel()
            return self.quantizer.get_quant_val(self.weight, max=wt_max)
        return self.weight

    def conv_integer(self, input):
        quant_wts = self.get_quant_weights()
        return F.conv2d(
            input, quant_wts, None, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        if self.quantize:
            wt_max = self._get_lim_per_channel()
            quant_wts = self.quantizer(self.weight, max=wt_max)
            return F.conv2d(
                input, quant_wts, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return F.conv2d(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class QRoundFn(enum.Enum):
    FLOOR = "floor"
    ROUND = "round"
    TRUNC = "trunc"


ROUND_FN = {QRoundFn.ROUND: QRound, QRoundFn.TRUNC: QTrunc, QRoundFn.FLOOR: QFloor}


def create_quantizer(
    quantize: bool,
    num_bits: int,
    trainable: bool,
    unsigned: bool,
    round_fn: QRoundFn,
    max_val: float,
) -> torch.nn.Module:
    quantizer_class = QuantUnsigned if unsigned else QuantSymmetric
    round_class = ROUND_FN[round_fn]
    quantizer = quantizer_class(num_bits, round_class)
    quant_class = TrainedQuant if trainable else FixedQuant
    return quant_class(quantize, quantizer, max_val)


# ============================================================================


class InitMethod(Enum):
    DEFAULT = "default"
    HE_UNIFORM = "he_unif"
    HE_NORMAL = "he_norm"


class ShortcutType(Enum):
    NONE = "none"
    CONCAT = "concat"
    ADD = "add"
    MAX = "max"


def _init_conv_weights(
    conv2d: torch.nn.Conv2d,
    activation: torch.nn.Module,
    init_method: InitMethod,
    init_icnr: bool = False,
):
    if init_method == InitMethod.HE_UNIFORM:
        init_fn = torch.nn.init.kaiming_uniform_
    elif init_method == InitMethod.HE_NORMAL:
        init_fn = torch.nn.init.kaiming_normal_
    else:
        init_fn = None

    if init_icnr:
        if init_fn is None:
            icnr(conv2d.weight, conv2d.bias)
        else:
            init_fn = partial(icnr, bias=conv2d.bias, initializer=init_fn)

    if init_fn is None:
        return

    if isinstance(activation, torch.nn.LeakyReLU):
        init_fn(conv2d.weight, a=activation.negative_slope, nonlinearity="leaky_relu")
    if isinstance(activation, torch.nn.PReLU):
        init_fn(conv2d.weight, a=activation.weight.item(), nonlinearity="leaky_relu")
    else:
        init_fn(conv2d.weight, nonlinearity="relu")


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        activation_type: ActivationType = ActivationType.RELU,
        signed_quant: bool = True,
        quantize_conv: bool = False,
        quantize_act: bool = False,
        activation_quantizer=None,
        integer_forward: bool = False,
        num_bits_conv: int = 8,
        num_bits_act: int = 8,
        padding_mode: str = "zeros",
        round_fn: QRoundFn = QRoundFn.TRUNC,
        init_method: InitMethod = InitMethod.HE_NORMAL,
        init_icnr: bool = False,
        **kwargs,
    ):
        super().__init__()

        if integer_forward and not quantize_conv:
            raise Exception("integer_forward is only valid for quantized conv operation.")

        self.out_channels = out_channels
        self.quantize_conv = quantize_conv
        self.integer_forward = integer_forward
        self.signed_quant = signed_quant
        self.activation_type = activation_type

        round_class = {QRoundFn.TRUNC: QTrunc, QRoundFn.FLOOR: QFloor, QRoundFn.ROUND: QRound}[
            round_fn
        ]

        self.blocks = {
            "activation": create_activation(activation_type),
            "convolution": QConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=1,
                padding_mode=padding_mode,
                bias=bias,
                quantize=quantize_conv,
                num_bits=num_bits_conv,
                round_fn=round_class,
            ),
            "activation_quantizer": activation_quantizer,
        }

        if activation_quantizer is None:
            self.blocks["activation_quantizer"] = create_quantizer(
                quantize=quantize_act,
                num_bits=num_bits_act,
                trainable=True,
                unsigned=not signed_quant,
                round_fn=round_fn,
                max_val={8: 3.0, 4: 0.5}[num_bits_act],
            )

        self.activation = self.blocks["activation"]
        self.act_quantizer = self.blocks["activation_quantizer"]
        self.conv2d = self.blocks["convolution"]
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

        _init_conv_weights(
            self.blocks["convolution"], self.blocks["activation"], init_method, init_icnr
        )

    def get_out_channels(self) -> int:
        return self.out_channels

    def get_act_quantizer(self):
        return self.blocks["activation_quantizer"]

    def set_act_quantizer(self, quantizer):
        self.blocks["activation_quantizer"] = quantizer

    def compute_params(self, input_step_size):
        if self.blocks["activation_quantizer"]:
            act_step = self.blocks["activation_quantizer"].get_quant_step()
        else:
            act_step = 1

        mac_bias = self.blocks["convolution"].bias / act_step
        wt_step = self.blocks["convolution"].get_quant_step().flatten()
        mac_scale = input_step_size * wt_step / act_step
        return act_step, mac_bias, mac_scale

    def _integer_forward_impl(self, x):
        input_step_size = x[1]
        x_q = self.blocks["convolution"].conv_integer(x[0])
        (act_step, mac_bias, mac_scale) = self.compute_params(input_step_size)
        y_q = x_q * mac_scale.view(1, -1, 1, 1) + mac_bias.view(1, -1, 1, 1)
        y_q = self.pool(y_q)
        if self.blocks["activation"]:
            y_q = self.blocks["activation"](y_q)
        if self.blocks["activation_quantizer"]:
            z_q = self.blocks["activation_quantizer"].integer(y_q)
            return z_q, act_step
        else:
            return y_q

    def _forward_impl(self, x):
        x = self.blocks["convolution"](x)
        if self.blocks["activation"]:
            x = self.blocks["activation"](x)
        if self.blocks["activation_quantizer"]:
            x = self.blocks["activation_quantizer"](x)
        x = self.pool(x)
        return x

    def forward(self, x):
        if self.integer_forward:
            return self._integer_forward_impl(x)
        return self._forward_impl(x)


Model = ConvBlock


@triton.jit
def fused_conv_relu_pool(
    x_ptr, w_ptr, b_ptr, o_ptr,
    H, W, OH, OW,
    BPH: tl.constexpr, BPW: tl.constexpr,
    CIN: tl.constexpr, COUT: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    ph0 = pid_h * BPH
    pw0 = pid_w * BPW

    p = tl.arange(0, BPH * BPW)
    ppi = p // BPW
    ppj = p % BPW
    poh = ph0 + ppi
    powc = pw0 + ppj
    pool_valid = (poh < OH) & (powc < OW)

    cin = tl.arange(0, CIN)
    cout = tl.arange(0, COUT)
    bias = tl.load(b_ptr + cout).to(tl.float32)

    HW = H * W

    acc00 = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
    acc01 = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
    acc10 = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
    acc11 = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)

    base_h = 2 * poh
    base_w = 2 * powc

    # Preload all 9 weight taps into registers (tiny: CIN x COUT x 9).
    w00 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 0)
    w01 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 1)
    w02 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 2)
    w03 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 3)
    w04 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 4)
    w05 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 5)
    w06 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 6)
    w07 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 7)
    w08 = tl.load(w_ptr + cout[None, :] * (CIN * 9) + cin[:, None] * 9 + 8)

    for r in range(4):
        ih = base_h - 1 + r
        rvalid = (ih >= 0) & (ih < H)
        for c in range(4):
            iw = base_w - 1 + c
            cvalid = (iw >= 0) & (iw < W)
            mvalid = rvalid & cvalid
            base = ih * W + iw
            xptrs = x_ptr + cin[None, :] * HW + base[:, None]
            xblk = tl.load(xptrs, mask=mvalid[:, None], other=0.0)

            for si in range(2):
                kh = r - si
                if 0 <= kh <= 2:
                    for sj in range(2):
                        kw = c - sj
                        if 0 <= kw <= 2:
                            tap = kh * 3 + kw
                            if tap == 0:
                                wblk = w00
                            elif tap == 1:
                                wblk = w01
                            elif tap == 2:
                                wblk = w02
                            elif tap == 3:
                                wblk = w03
                            elif tap == 4:
                                wblk = w04
                            elif tap == 5:
                                wblk = w05
                            elif tap == 6:
                                wblk = w06
                            elif tap == 7:
                                wblk = w07
                            else:
                                wblk = w08
                            d = tl.dot(xblk, wblk, out_dtype=tl.float32)
                            if si == 0 and sj == 0:
                                acc00 += d
                            elif si == 0 and sj == 1:
                                acc01 += d
                            elif si == 1 and sj == 0:
                                acc10 += d
                            else:
                                acc11 += d

    acc00 = tl.maximum(acc00 + bias[None, :], 0.0)
    acc01 = tl.maximum(acc01 + bias[None, :], 0.0)
    acc10 = tl.maximum(acc10 + bias[None, :], 0.0)
    acc11 = tl.maximum(acc11 + bias[None, :], 0.0)
    pooled = (acc00 + acc01 + acc10 + acc11) * 0.25

    optrs = o_ptr + cout[None, :] * (OH * OW) + (poh * OW + powc)[:, None]
    tl.store(optrs, pooled.to(tl.float16), mask=pool_valid[:, None])


class ModelNew(ConvBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fused_forward(self, x):
        x = x.contiguous()
        N, Cin, H, W = x.shape
        assert N == 1

        w = self.conv2d.weight.contiguous()
        Cout = w.shape[0]
        assert w.shape[2] == 3 and w.shape[3] == 3

        if self.conv2d.bias is not None:
            b = self.conv2d.bias.contiguous()
        else:
            b = torch.zeros(Cout, dtype=x.dtype, device=x.device)

        OH = H // 2
        OW = W // 2
        out = torch.empty((1, Cout, OH, OW), dtype=x.dtype, device=x.device)

        BPH = 8
        BPW = 64
        grid = (triton.cdiv(OH, BPH), triton.cdiv(OW, BPW))

        fused_conv_relu_pool[grid](
            x, w, b, out,
            H, W, OH, OW,
            BPH=BPH, BPW=BPW,
            CIN=Cin, COUT=Cout,
            num_warps=8,
        )
        return out

    def forward(self, x):
        if self.integer_forward:
            return self._integer_forward_impl(x)

        if (not self.quantize_conv) and (self.act_quantizer is None) \
                and self.activation_type == ActivationType.RELU \
                and self.conv2d.kernel_size == (3, 3) \
                and self.conv2d.stride == (1, 1):
            return self._fused_forward(x)

        return self._forward_impl(x)


def get_inputs():
    # Half-precision inputs to match the half-precision Model.
    return [torch.randn(1, 16, 720, 1280, dtype=torch.float16)]


def get_init_inputs():
    in_channels = 16
    out_channels = 32
    quantize = False
    integer_forward = False
    return [
        in_channels,              # in_channels
        out_channels,             # out_channels
        3,                        # kernel_size
        1,                        # stride (default)
        True,                     # bias (default)
        ActivationType.RELU,      # activation_type
        True,                     # signed_quant (default)
        quantize,                 # quantize_conv
        quantize,                 # quantize_act
        None,                     # activation_quantizer
        integer_forward,          # integer_forward
        8,                        # num_bits_conv
        8,                        # num_bits_act
        "zeros",                  # padding_mode
    ]