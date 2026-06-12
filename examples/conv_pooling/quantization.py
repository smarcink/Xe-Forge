import enum
from dataclasses import dataclass
from typing import Optional, Self, Type

import torch as torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_QUANT_THRESHOLD = 3.0


@dataclass
class QTensor:
    data: torch.Tensor
    scale: torch.Tensor

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    def __iter__(self):
        # allows: data, scale = qt
        yield self.data
        yield self.scale

    def __getitem__(self, idx: int):
        # allows: qt[0] -> data, qt[1] -> scale
        if idx == 0:
            return self.data
        if idx == 1:
            return self.scale
        raise IndexError(f"QTensor only supports indices 0 (data) and 1 (scale), got {idx}")

    def __add__(self, other: "QTensor") -> Self:
        # this simple addition is only valid for int8 and with matching scale factors (the same quantizers used)
        if not torch.equal(self.scale, other.scale):
            raise ValueError("Cannot add QTensors with different scales")
        # ToDo: find a better way to specify these params
        num_bits = 8
        levels = 2**num_bits
        qmin = -levels // 2
        qmax = levels // 2 - 1
        # saturated int8 addition
        result = (self.data + other.data).clamp(qmin, qmax)
        return QTensor(data=result, scale=self.scale)

    def permute(self, *dims) -> Self:
        """Permute only the underlying data tensor; keep scale unchanged."""
        return QTensor(data=self.data.permute(*dims), scale=self.scale)

    def contiguous(self, memory_format: Optional[torch.memory_format] = None) -> Self:
        """Make only the underlying data tensor contiguous; keep scale unchanged."""
        if memory_format is None:
            return QTensor(data=self.data.contiguous(), scale=self.scale)
        return QTensor(data=self.data.contiguous(memory_format=memory_format), scale=self.scale)

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert QTensor back to a FP tensor."""
        y = self.data
        if dtype is None:
            dtype = torch.float32
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_dtype(y.device.type)
        return (y * self.scale).to(dtype)


def dequantize_tensor(x) -> torch.Tensor:
    if isinstance(x, QTensor):
        return x.dequantize()
    return x


class QFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class QRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class QTrunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # TODO: why trunc instead of round ?
        return torch.trunc(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class QFP8Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # snap higher precision values to FP8 grid in forward
        return input.to(dtype=torch.float8_e4m3fn).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class QuantFP8(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8_dtype = torch.float8_e4m3fn
        dt_info = torch.finfo(self.fp8_dtype)
        self.qmin = float(dt_info.min)
        self.qmax = float(dt_info.max)

    def get_quant_step(self, max: torch.Tensor) -> torch.Tensor:
        return max / self.qmax

    def get_quant_val(self, x: torch.Tensor, max: torch.Tensor) -> torch.Tensor:
        scale = self.get_quant_step(max)
        # Normalize
        y = x / scale
        # Saturate to FP8 finite range
        y = y.clamp(min=self.qmin, max=self.qmax)
        # Cast/round
        y = QFP8Round.apply(y)
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, max: torch.Tensor) -> torch.Tensor:
        step = self.get_quant_step(max.float())
        xq = self.get_quant_val(x.float(), max=max.float())
        # dequantize
        return (xq * step).to(x.dtype)


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
        xq = self.qround(x)
        return self._clamp(xq)

    def get_quant_step(self, max):
        return max * self._scale

    def get_quant_val(self, x, max):
        step = self.get_quant_step(max)
        return self._quantize(x, step)

    def forward(self, x, max):
        step = self.get_quant_step(max.float())
        xq = self._quantize(x.float(), step)
        # dequantize
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
        xq = self.qround(x)
        return self._clamp(xq)

    def get_quant_step(self, max):
        return 2 * max * self.scale

    def get_quant_val(self, x, max):
        step = self.get_quant_step(max)
        return self._quantize(x, step)

    def forward(self, x, max):
        step = self.get_quant_step(max.float())
        xq = self._quantize(x.float(), step)
        # dequantize
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
        else:
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

    def get_quant_step(self):
        if not self.quantize:
            raise Exception("Quantization is not enabled")
        th_max = self._get_threshold()
        return self.quantizer.get_quant_step(th_max)

    def _get_threshold(self):
        return torch.exp(self.max_log)

    def integer(self, x):
        return self.quantizer.integer(x)

    def forward(self, x):
        if self.quantize:
            th_max = self._get_threshold()
            return self.quantizer(x, max=th_max)

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
        wt_max = self._get_lim_per_channel()
        return self.quantizer.get_quant_step(wt_max)

    def get_quant_weights(self):
        if self.quantize:
            wt_max = self._get_lim_per_channel()
            return self.quantizer.get_quant_val(self.weight, max=wt_max)

        return self.weight

    # quantized integer convolution
    def conv_integer(self, input):
        quant_wts = self.get_quant_weights()
        return F.conv2d(
            input, quant_wts, None, self.stride, self.padding, self.dilation, self.groups
        )

    # simulated quantization
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


class SmoothQuant(nn.Module):
    """Learnable per-channel activation smoother for SmoothQuant.

    Divides activations by per-channel factors before quantization.
    The same factors are absorbed into the next layer's weights,
    preserving the mathematical identity in FP:
        y = (x / diag(s)) @ (W * diag(s))^T = x @ W^T

    Calibration phase observes per-channel activation magnitudes via EMA,
    then initializes smooth_log = log(running_ch_max) so the initial
    smooth factors equal the observed per-channel max. This makes
    x / s ≈ uniform across channels after calibration.
    """

    def __init__(self, num_channels: int, quantize: bool = False):
        super().__init__()
        self.quantize = quantize
        self.num_channels = num_channels
        self.smooth_log = nn.Parameter(torch.zeros(num_channels), requires_grad=False)
        # Per-channel EMA observer
        self._obs_enabled = False
        self._obs_beta = 0.01
        self.register_buffer("_running_ch_max", torch.ones(num_channels))
        self.register_buffer("_obs_initialized", torch.tensor(False))

    def start_observing(self) -> None:
        self._obs_enabled = True

    @torch.compiler.disable
    def _observe_impl(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            # Collapse all dims except last (channel) to get per-channel abs max
            ch_abs_max = x.detach().abs().amax(dim=tuple(range(x.ndim - 1)))
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(ch_abs_max, op=dist.ReduceOp.MAX)
            if not bool(self._obs_initialized.item()):
                self._running_ch_max.copy_(ch_abs_max)
                self._obs_initialized.fill_(True)
            else:
                self._running_ch_max.mul_(1.0 - self._obs_beta).add_(
                    ch_abs_max, alpha=self._obs_beta
                )

    def stop_observing(self) -> None:
        if not self._obs_enabled:
            raise RuntimeError("SmoothQuant calibration was not enabled!")
        self._obs_enabled = False
        # Initialize smooth_log from observed per-channel max
        ch_max = self._running_ch_max.detach().clamp(min=1e-8)
        self.smooth_log.data = torch.log(ch_max).to(self.smooth_log.dtype)
        self.smooth_log.requires_grad_(True)

    @property
    def smooth_factors(self) -> torch.Tensor:
        return torch.exp(self.smooth_log.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._obs_enabled:
            self._observe_impl(x)
        if not self.quantize:
            return x
        return (x.float() / self.smooth_factors).to(x.dtype)


class QLinear(nn.Linear):
    def __init__(self, quantize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = quantize
        self.quantizer = QuantSymmetric(num_bits=8, qround=QTrunc)
        self._weight_smooth: SmoothQuant | None = None

    def _effective_weight(self) -> torch.Tensor:
        """Return weight with smooth factors absorbed: W * diag(s)."""
        if self._weight_smooth is not None and self._weight_smooth.quantize:
            return self.weight * self._weight_smooth.smooth_factors.unsqueeze(0)
        return self.weight

    def _get_lim_per_channel(self):
        return self._effective_weight().abs().amax(dim=1, keepdim=True)

    def get_quant_step(self):
        wt_max = self._get_lim_per_channel()
        return self.quantizer.get_quant_step(wt_max)

    def get_quant_weights(self):
        if self.quantize:
            w = self._effective_weight()
            wt_max = w.abs().amax(dim=1, keepdim=True)
            return self.quantizer.get_quant_val(w, max=wt_max)

        return self._effective_weight()

    def linear_integer(self, input):
        quant_wts = self.get_quant_weights().to(dtype=input.dtype)
        return F.linear(input, quant_wts)

    def forward(self, x):
        if self.quantize:
            w = self._effective_weight()
            wt_max = w.abs().amax(dim=1, keepdim=True)
            quant_wts = self.quantizer(w, max=wt_max)
            return F.linear(x, quant_wts, self.bias)
        return F.linear(x, self._effective_weight(), self.bias)


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
    # Always create a 'quant' object for a quantizer. However, the behaviour of the quantizer is controled by the 'quantize' flag,
    # which determines whether the operation should actually quantize the input tensor or simply return it unchanged.
    # This design ensures that both floating-point and quantized models have identical state dictionaries,
    # making it easier to load checkpoints during quantization-aware training (QAT) without introducing unnecessary complications.
    quantizer_class = QuantUnsigned if unsigned else QuantSymmetric
    round_class = ROUND_FN[round_fn]
    quantizer = quantizer_class(num_bits, round_class)

    quant_class = TrainedQuant if trainable else FixedQuant
    return quant_class(quantize, quantizer, max_val)


class StatsObserver(nn.Module):
    def __init__(self, beta: float = 0.01):
        super().__init__()
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"StatsObserver EMA beta must be in (0, 1], got {beta}")
        self.beta = beta
        self.enabled = False
        self.register_buffer("running_abs_max", torch.tensor([1.0]))  # EMA buffer
        self.register_buffer("_ema_initialized", torch.tensor(False))

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        if not self.enabled:
            raise RuntimeError("Calibration was not enabled!")
        self.enabled = False

    def _sync_ddp(self, local_max: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        return local_max

    @torch.compiler.disable
    def _observe_impl(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            local_abs_max = x.detach().abs().amax()

            local_abs_max = local_abs_max.view_as(self.running_abs_max)
            global_abs_max = self._sync_ddp(local_abs_max)
            if not bool(self._ema_initialized.item()):
                self.running_abs_max.copy_(global_abs_max)
                self._ema_initialized.fill_(True)
            else:
                # EMA update
                self.running_abs_max.mul_(1.0 - self.beta).add_(global_abs_max, alpha=self.beta)

    def observe(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        self._observe_impl(x)


class TrainedCalibratedQuant(nn.Module):
    def __init__(self, quantize=False, integer_forward=False, quantizer=None):
        super().__init__()
        self.quantize = quantize
        self.quantizer = quantizer
        self.integer_forward = integer_forward
        self.observer = StatsObserver()
        # NOTE: max_log starts with requires_grad=False and is later enabled via
        # stop_observing(). This works because configure_optimizers() uses
        # model.parameters() which returns all nn.Parameter objects regardless of
        # requires_grad.
        self.max_log = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

    def start_observing(self) -> None:
        self.observer.start()

    def stop_observing(self) -> None:
        self.observer.stop()
        # take absolute max
        th = self.observer.running_abs_max.detach()
        # avoid log(0)
        th = torch.clamp(th, min=torch.tensor(1e-8, device=th.device, dtype=th.dtype))
        self.max_log.data = torch.log(th).to(self.max_log.dtype)
        self.max_log.requires_grad_(True)

    def get_quant_step(self):
        if not (self.quantize or self.integer_forward):
            raise Exception("Quantization is not enabled")
        th_max = self._get_threshold()
        return self.quantizer.get_quant_step(th_max)

    def integer(self, x):
        return self.quantizer.integer(x)

    def _get_threshold(self):
        return torch.exp(self.max_log.to(dtype=torch.float32))

    def forward(self, x):
        # Fast path: skip @torch.compiler.disable when not quantizing and not observing
        if not self.quantize and not self.observer.enabled:
            return x
        self.observer.observe(x)
        if self.quantize:
            if self.integer_forward:
                return QTensor(
                    self.quantizer.get_quant_val(x, self._get_threshold()).to(x.dtype),
                    self.get_quant_step().to(x.dtype),
                )
            else:
                return self.quantizer(x, max=self._get_threshold())
        return x
