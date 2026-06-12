import enum
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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


class QRoundFn(enum.Enum):
    FLOOR = "floor"
    ROUND = "round"
    TRUNC = "trunc"


class InitMethod(Enum):
    DEFAULT = "default"
    HE_UNIFORM = "he_unif"
    HE_NORMAL = "he_norm"


class ShortcutType(Enum):
    NONE = "none"
    CONCAT = "concat"
    ADD = "add"
    MAX = "max"


def create_activation(atype: ActivationType, inplace: bool = False, **kwargs) -> nn.Module:
    if atype == ActivationType.NONE:
        return nn.Identity()
    if atype == ActivationType.ELU:
        return nn.ELU(alpha=kwargs.get("elu_alpha", 1.0), inplace=inplace)
    if atype == ActivationType.RELU:
        return nn.ReLU(inplace=inplace)
    if atype == ActivationType.RELU6:
        return nn.ReLU6(inplace=inplace)
    if atype == ActivationType.LEAKY_RELU:
        return nn.LeakyReLU(negative_slope=kwargs.get("leaky_relu_slope", 0.02), inplace=inplace)
    if atype == ActivationType.PARAMETRIC_RELU:
        return nn.PReLU(init=kwargs.get("prelu_init", 0.25))
    if atype == ActivationType.SIGMOID:
        return nn.Sigmoid()
    if atype == ActivationType.GELU:
        return nn.GELU()
    if atype == ActivationType.HARDSIGMOID:
        return nn.Hardsigmoid()
    if atype == ActivationType.CLIP:
        return nn.Hardtanh(min_val=kwargs.get("clip_min", 0.0), max_val=kwargs.get("clip_max", 1.0))
    return nn.Identity()


class QConv2D(nn.Conv2d):
    def __init__(self, *args, quantize=False, num_bits=8, round_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = quantize

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _NoOpQuant(nn.Module):
    quantize = False

    def forward(self, x):
        return x

    def integer(self, x):
        return x

    def get_quant_step(self):
        return 1


def create_quantizer(*args, **kwargs):
    return _NoOpQuant()


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
        self.out_channels = out_channels
        self.quantize_conv = quantize_conv
        self.integer_forward = integer_forward
        self.signed_quant = signed_quant
        self.activation_type = activation_type

        self.activation = create_activation(activation_type)
        self.act_quantizer = activation_quantizer if activation_quantizer is not None else create_quantizer()
        self.conv2d = QConv2D(
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
            round_fn=round_fn,
        )
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

    def _integer_forward_impl(self, x):
        return self._forward_impl(x[0] if isinstance(x, (tuple, list)) else x)

    def _forward_impl(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        x = self.act_quantizer(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        if self.integer_forward:
            return self._integer_forward_impl(x)
        return self._forward_impl(x)


def _interior_autotune_configs():
    # BPH*BPW is the spatial tile size.  The original hand-picked tile was
    # 4x32 with 8 warps; this sweep includes smaller/larger spatial tiles and
    # at least one 32-warp large tile for Intel XPU.
    return [
        triton.Config({"BPH": 2, "BPW": 32}, num_warps=4, num_stages=2),
        triton.Config({"BPH": 2, "BPW": 32}, num_warps=4, num_stages=3),
        triton.Config({"BPH": 4, "BPW": 16}, num_warps=4, num_stages=3),
        triton.Config({"BPH": 4, "BPW": 32}, num_warps=8, num_stages=2),
        triton.Config({"BPH": 4, "BPW": 32}, num_warps=8, num_stages=3),
        triton.Config({"BPH": 8, "BPW": 16}, num_warps=8, num_stages=3),
        triton.Config({"BPH": 8, "BPW": 32}, num_warps=16, num_stages=3),
        triton.Config({"BPH": 16, "BPW": 16}, num_warps=16, num_stages=3),
        triton.Config({"BPH": 8, "BPW": 32}, num_warps=32, num_stages=3),
        triton.Config({"BPH": 16, "BPW": 16}, num_warps=32, num_stages=3),
    ]


def _cleanup_autotune_configs():
    return [
        triton.Config({"BLOCK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 128}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=32, num_stages=3),
    ]


@triton.autotune(
    configs=_interior_autotune_configs(),
    key=["H", "W", "CIN", "COUT", "HAS_BIAS"],
)
@triton.jit
def fused_conv_relu_pool_interior_nomask(
    x_ptr, w_ptr, b_ptr, o_ptr,
    H: tl.constexpr, W: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    MAIN_H: tl.constexpr, MAIN_W: tl.constexpr,
    BPH: tl.constexpr, BPW: tl.constexpr,
    CIN: tl.constexpr, COUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    ph0 = 1 + pid_h * BPH
    pw0 = 1 + pid_w * BPW

    p = tl.arange(0, BPH * BPW)
    ppi = p // BPW
    ppj = p - ppi * BPW
    poh = ph0 + ppi
    powc = pw0 + ppj

    cin = tl.arange(0, CIN)
    cout = tl.arange(0, COUT)

    if HAS_BIAS:
        bias = tl.load(b_ptr + cout).to(tl.float32)
    else:
        bias = tl.zeros((COUT,), dtype=tl.float32)

    pooled = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
    base_h = 2 * poh
    base_w = 2 * powc

    for si in range(2):
        acc_l = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
        acc_r = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
        oh_in = base_h + si

        for kh in range(3):
            ih = oh_in + kh - 1
            for cc in range(4):
                iw = base_w + cc - 1
                xoffs = ((ih * W + iw) * CIN)[:, None] + cin[None, :]
                xblk = tl.load(x_ptr + xoffs)

                if cc < 3:
                    tap_l = kh * 3 + cc
                    woffs_l = tap_l * (CIN * COUT) + cin[:, None] * COUT + cout[None, :]
                    wblk_l = tl.load(w_ptr + woffs_l)
                    acc_l += tl.dot(xblk, wblk_l, out_dtype=tl.float32)

                if cc > 0:
                    tap_r = kh * 3 + (cc - 1)
                    woffs_r = tap_r * (CIN * COUT) + cin[:, None] * COUT + cout[None, :]
                    wblk_r = tl.load(w_ptr + woffs_r)
                    acc_r += tl.dot(xblk, wblk_r, out_dtype=tl.float32)

        pooled += tl.maximum(acc_l + bias[None, :], 0.0)
        pooled += tl.maximum(acc_r + bias[None, :], 0.0)

    pooled *= 0.25
    ooffs = ((poh * OW + powc) * COUT)[:, None] + cout[None, :]
    tl.store(o_ptr + ooffs, pooled.to(tl.float16))


@triton.autotune(
    configs=_cleanup_autotune_configs(),
    key=["H", "W", "CIN", "COUT", "TOTAL", "HAS_BIAS"],
)
@triton.jit
def fused_conv_relu_pool_cleanup(
    x_ptr, w_ptr, b_ptr, o_ptr,
    H: tl.constexpr, W: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    MAIN_H: tl.constexpr, MAIN_W: tl.constexpr,
    TOTAL: tl.constexpr, CLEAN_FULL: tl.constexpr, CLEAN_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    CIN: tl.constexpr, COUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < TOTAL

    full_part = CLEAN_FULL * OW
    in_full = offs < full_part
    rblk = offs // OW
    cfull = offs - rblk * OW
    row_full = tl.where(rblk == 0, 0, MAIN_H + rblk)

    rem = offs - full_part
    rmid = rem // CLEAN_COLS
    cmid_idx = rem - rmid * CLEAN_COLS
    row_mid = rmid + 1
    col_mid = tl.where(cmid_idx == 0, 0, MAIN_W + cmid_idx)

    poh = tl.where(in_full, row_full, row_mid)
    powc = tl.where(in_full, cfull, col_mid)

    cin = tl.arange(0, CIN)
    cout = tl.arange(0, COUT)

    if HAS_BIAS:
        bias = tl.load(b_ptr + cout).to(tl.float32)
    else:
        bias = tl.zeros((COUT,), dtype=tl.float32)

    pooled = tl.zeros((BLOCK, COUT), dtype=tl.float32)
    base_h = 2 * poh
    base_w = 2 * powc

    for si in range(2):
        for sj in range(2):
            acc = tl.zeros((BLOCK, COUT), dtype=tl.float32)
            oh_in = base_h + si
            ow_in = base_w + sj
            for kh in range(3):
                ih = oh_in + kh - 1
                rvalid = (ih >= 0) & (ih < H) & valid
                for kw in range(3):
                    iw = ow_in + kw - 1
                    mvalid = rvalid & (iw >= 0) & (iw < W)
                    xoffs = ((ih * W + iw) * CIN)[:, None] + cin[None, :]
                    xblk = tl.load(x_ptr + xoffs, mask=mvalid[:, None], other=0.0)
                    tap = kh * 3 + kw
                    woffs = tap * (CIN * COUT) + cin[:, None] * COUT + cout[None, :]
                    wblk = tl.load(w_ptr + woffs)
                    acc += tl.dot(xblk, wblk, out_dtype=tl.float32)
            pooled += tl.maximum(acc + bias[None, :], 0.0)

    pooled *= 0.25
    ooffs = ((poh * OW + powc) * COUT)[:, None] + cout[None, :]
    tl.store(o_ptr + ooffs, pooled.to(tl.float16), mask=valid[:, None])


@triton.autotune(
    configs=_interior_autotune_configs(),
    key=["H", "W", "CIN", "COUT", "HAS_BIAS"],
)
@triton.jit
def fused_conv_relu_pool_hwio_nhwc_hpair(
    x_ptr, w_ptr, b_ptr, o_ptr,
    H: tl.constexpr, W: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    BPH: tl.constexpr, BPW: tl.constexpr,
    CIN: tl.constexpr, COUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    ph0 = pid_h * BPH
    pw0 = pid_w * BPW

    p = tl.arange(0, BPH * BPW)
    ppi = p // BPW
    ppj = p - ppi * BPW
    poh = ph0 + ppi
    powc = pw0 + ppj
    pool_valid = (poh < OH) & (powc < OW)

    cin = tl.arange(0, CIN)
    cout = tl.arange(0, COUT)

    if HAS_BIAS:
        bias = tl.load(b_ptr + cout).to(tl.float32)
    else:
        bias = tl.zeros((COUT,), dtype=tl.float32)

    pooled = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
    base_h = 2 * poh
    base_w = 2 * powc

    for si in range(2):
        acc_l = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
        acc_r = tl.zeros((BPH * BPW, COUT), dtype=tl.float32)
        oh_in = base_h + si

        for kh in range(3):
            ih = oh_in + kh - 1
            rvalid = (ih >= 0) & (ih < H)
            for cc in range(4):
                iw = base_w + cc - 1
                cvalid = (iw >= 0) & (iw < W)
                mvalid = rvalid & cvalid & pool_valid
                xoffs = ((ih * W + iw) * CIN)[:, None] + cin[None, :]
                xblk = tl.load(x_ptr + xoffs, mask=mvalid[:, None], other=0.0)

                if cc < 3:
                    tap_l = kh * 3 + cc
                    woffs_l = tap_l * (CIN * COUT) + cin[:, None] * COUT + cout[None, :]
                    wblk_l = tl.load(w_ptr + woffs_l)
                    acc_l += tl.dot(xblk, wblk_l, out_dtype=tl.float32)

                if cc > 0:
                    tap_r = kh * 3 + (cc - 1)
                    woffs_r = tap_r * (CIN * COUT) + cin[:, None] * COUT + cout[None, :]
                    wblk_r = tl.load(w_ptr + woffs_r)
                    acc_r += tl.dot(xblk, wblk_r, out_dtype=tl.float32)

        pooled += tl.maximum(acc_l + bias[None, :], 0.0)
        pooled += tl.maximum(acc_r + bias[None, :], 0.0)

    pooled *= 0.25
    ooffs = ((poh * OW + powc) * COUT)[:, None] + cout[None, :]
    tl.store(o_ptr + ooffs, pooled.to(tl.float16), mask=pool_valid[:, None])


class ModelNew(ConvBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._w_packed = None
        self._w_packed_key = None
        self._zero_bias = None
        self._zero_bias_key = None

    def _act_quant_is_noop(self):
        q = self.act_quantizer
        return q is None or (hasattr(q, "quantize") and not bool(q.quantize))

    def _ensure_xpu_param_dtype(self, device, dtype):
        w = self.conv2d.weight
        if w.device != device or w.dtype != dtype or not w.is_contiguous():
            self.conv2d.weight.data = w.detach().to(device=device, dtype=dtype).contiguous()
            self._w_packed = None
            self._w_packed_key = None

        if self.conv2d.bias is not None:
            b = self.conv2d.bias
            if b.device != device or b.dtype != dtype or not b.is_contiguous():
                self.conv2d.bias.data = b.detach().to(device=device, dtype=dtype).contiguous()

    def _ensure_packed_weight(self, device, dtype):
        self._ensure_xpu_param_dtype(device, dtype)
        w = self.conv2d.weight
        key = (int(w._version), w.data_ptr(), str(w.device), str(w.dtype))
        if self._w_packed is None or self._w_packed_key != key:
            self._w_packed = w.detach().permute(2, 3, 1, 0).contiguous()
            self._w_packed_key = key
        return self._w_packed

    def _get_zero_bias(self, device, dtype, cout):
        key = (str(device), str(dtype), int(cout))
        if self._zero_bias is None or self._zero_bias_key != key:
            self._zero_bias = torch.zeros((cout,), device=device, dtype=dtype)
            self._zero_bias_key = key
        return self._zero_bias

    def _fused_forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to(device="xpu", dtype=torch.float16, memory_format=torch.channels_last)
        else:
            x = x.contiguous(memory_format=torch.channels_last)

        N, Cin, H, W = x.shape
        assert N == 1

        w_packed = self._ensure_packed_weight(x.device, x.dtype)
        Cout = self.conv2d.weight.shape[0]

        has_bias = self.conv2d.bias is not None
        if has_bias:
            b = self.conv2d.bias.contiguous()
        else:
            b = self._get_zero_bias(x.device, x.dtype, Cout)

        OH = H // 2
        OW = W // 2
        out = torch.empty((1, Cout, OH, OW), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)

        # Autotuned BPH values divide 16 and BPW values divide 32, so the
        # mask-free interior size is valid for every candidate configuration.
        MAIN_BPH_ALIGN = 16
        MAIN_BPW_ALIGN = 32
        main_h = ((OH - 2) // MAIN_BPH_ALIGN) * MAIN_BPH_ALIGN
        main_w = ((OW - 2) // MAIN_BPW_ALIGN) * MAIN_BPW_ALIGN

        if main_h > 0 and main_w > 0:
            grid_main = lambda meta: (main_h // meta["BPH"], main_w // meta["BPW"])
            fused_conv_relu_pool_interior_nomask[grid_main](
                x, w_packed, b, out,
                H, W, OH, OW,
                main_h, main_w,
                CIN=Cin, COUT=Cout,
                HAS_BIAS=has_bias,
                grf_mode="auto",
            )

            clean_full = OH - main_h
            clean_cols = OW - main_w
            total_clean = clean_full * OW + main_h * clean_cols

            grid_clean = lambda meta: (triton.cdiv(total_clean, meta["BLOCK"]),)
            fused_conv_relu_pool_cleanup[grid_clean](
                x, w_packed, b, out,
                H, W, OH, OW,
                main_h, main_w,
                total_clean, clean_full, clean_cols,
                CIN=Cin, COUT=Cout,
                HAS_BIAS=has_bias,
                grf_mode="auto",
            )
        else:
            grid = lambda meta: (triton.cdiv(OH, meta["BPH"]), triton.cdiv(OW, meta["BPW"]))
            fused_conv_relu_pool_hwio_nhwc_hpair[grid](
                x, w_packed, b, out,
                H, W, OH, OW,
                CIN=Cin, COUT=Cout,
                HAS_BIAS=has_bias,
                grf_mode="auto",
            )

        return out

    def forward(self, x):
        if self.integer_forward:
            return self._integer_forward_impl(x)

        if (
            (not self.quantize_conv)
            and self._act_quant_is_noop()
            and self.activation_type == ActivationType.RELU
            and self.conv2d.kernel_size == (3, 3)
            and self.conv2d.stride == (1, 1)
            and self.conv2d.padding == (1, 1)
            and self.conv2d.dilation == (1, 1)
            and self.conv2d.groups == 1
            and self.conv2d.padding_mode == "zeros"
        ):
            return self._fused_forward(x)

        return self._forward_impl(x)


Model = ModelNew


def get_inputs():
    return [torch.randn(1, 16, 720, 1280, dtype=torch.float16)]


def get_init_inputs():
    in_channels = 16
    out_channels = 32
    quantize = False
    integer_forward = False
    return [
        in_channels,
        out_channels,
        3,
        1,
        True,
        ActivationType.RELU,
        True,
        quantize,
        quantize,
        None,
        integer_forward,
        8,
        8,
        "zeros",
    ]