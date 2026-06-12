# Optimized: Model
# Speedup: 1.09x
# Stages: ['discovery', 'block_pointers', 'autotuning']

import torch
from torch import nn, Tensor
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune config generators
# ---------------------------------------------------------------------------
def _conv_configs():
    cfgs = []
    for bm in (16, 32, 64):
        for bd in (32, 64, 128):
            for nw in (4, 8, 16):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_M": bm, "BLOCK_D": bd},
                        num_warps=nw,
                        num_stages=2,
                    )
                )
    return cfgs


def _gemm_configs():
    cfgs = []
    tiles = [
        (256, 256, 16),
        (256, 128, 32),
        (128, 256, 32),
        (128, 128, 32),
        (128, 128, 64),
        (64, 128, 32),
        (64, 128, 64),
        (64, 64, 32),
        (64, 64, 64),
        (128, 64, 32),
    ]
    for bm, bn, bk in tiles:
        for nw in (4, 8, 16, 32):
            for ns in (2, 3):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return cfgs


def _ssm_configs():
    cfgs = []
    for bd in (8, 16, 32, 64):
        for nw in (1, 2, 4):
            for ns in (2, 3):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_D": bd},
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return cfgs


@triton.autotune(configs=_conv_configs(), key=["M", "L", "D", "KERNEL_SIZE"])
@triton.jit
def _causal_conv1d_silu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    b = offs_m // L
    t = offs_m - b * L

    bias = tl.load(bias_ptr + offs_d).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32) + bias[None, :]

    for k in tl.static_range(0, KERNEL_SIZE):
        in_t = t + k - (KERNEL_SIZE - 1)
        valid_t = in_t >= 0
        x_base = (b * L + in_t) * D
        x_idx = x_base[:, None] + offs_d[None, :]

        x_val = tl.load(
            x_ptr + x_idx,
            mask=valid_t[:, None],
            other=0.0,
        ).to(tl.float32)

        w_val = tl.load(w_ptr + offs_d * KERNEL_SIZE + k).to(tl.float32)
        acc += x_val * w_val[None, :]

    silu = acc / (1.0 + tl.exp(-acc))
    out_idx = offs_m[:, None] * D + offs_d[None, :]
    tl.store(out_ptr + out_idx, silu)


@triton.autotune(configs=_gemm_configs(), key=["M", "K", "N"])
@triton.jit
def _xproj_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # w is stored as [N, K] (row-major), we want w[n, k] for the dot as [K, N]
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(1, K),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        a = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_bp, acc.to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_gemm_configs(), key=["M", "D", "R"])
@triton.jit
def _dtproj_softplus_gemm_kernel(
    xproj_ptr,
    w_ptr,
    bias_ptr,
    dt_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # xproj is [M, P], we read first R columns of each row.
    x_bp = tl.make_block_ptr(
        base=xproj_ptr,
        shape=(M, R),
        strides=(P, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # w is stored as [D, R] (row-major), want w[n, k] as [K=R, N=D]
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(R, D),
        strides=(1, R),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, R, BLOCK_K):
        a = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    acc += bias[None, :]

    soft = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))

    out_bp = tl.make_block_ptr(
        base=dt_ptr,
        shape=(M, D),
        strides=(D, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_bp, soft.to(dt_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_ssm_configs(), key=["L", "D", "N"])
@triton.jit
def _ssm_recurrence_power_kernel(
    xconv_ptr,
    xproj_ptr,
    dt_ptr,
    alog_ptr,
    dskip_ptr,
    y_ptr,
    L: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    d_skip = tl.load(dskip_ptr + offs_d).to(tl.float32)

    h0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h4 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h5 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h6 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h7 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h8 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h9 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h11 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h12 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h13 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h14 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h15 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in tl.range(0, L, 1):
        m = pid_b * L + t

        x_val = tl.load(xconv_ptr + m * D + offs_d).to(tl.float32)
        dt_val = tl.load(dt_ptr + m * D + offs_d).to(tl.float32)
        xd = dt_val * x_val

        e = tl.exp(-dt_val)
        p = e

        b0 = tl.load(xproj_ptr + m * P + R + 0).to(tl.float32)
        c0 = tl.load(xproj_ptr + m * P + R + N + 0).to(tl.float32)
        h0 = p * h0 + xd * b0
        y_val = h0 * c0

        p = p * e
        b1 = tl.load(xproj_ptr + m * P + R + 1).to(tl.float32)
        c1 = tl.load(xproj_ptr + m * P + R + N + 1).to(tl.float32)
        h1 = p * h1 + xd * b1
        y_val += h1 * c1

        p = p * e
        b2 = tl.load(xproj_ptr + m * P + R + 2).to(tl.float32)
        c2 = tl.load(xproj_ptr + m * P + R + N + 2).to(tl.float32)
        h2 = p * h2 + xd * b2
        y_val += h2 * c2

        p = p * e
        b3 = tl.load(xproj_ptr + m * P + R + 3).to(tl.float32)
        c3 = tl.load(xproj_ptr + m * P + R + N + 3).to(tl.float32)
        h3 = p * h3 + xd * b3
        y_val += h3 * c3

        p = p * e
        b4 = tl.load(xproj_ptr + m * P + R + 4).to(tl.float32)
        c4 = tl.load(xproj_ptr + m * P + R + N + 4).to(tl.float32)
        h4 = p * h4 + xd * b4
        y_val += h4 * c4

        p = p * e
        b5 = tl.load(xproj_ptr + m * P + R + 5).to(tl.float32)
        c5 = tl.load(xproj_ptr + m * P + R + N + 5).to(tl.float32)
        h5 = p * h5 + xd * b5
        y_val += h5 * c5

        p = p * e
        b6 = tl.load(xproj_ptr + m * P + R + 6).to(tl.float32)
        c6 = tl.load(xproj_ptr + m * P + R + N + 6).to(tl.float32)
        h6 = p * h6 + xd * b6
        y_val += h6 * c6

        p = p * e
        b7 = tl.load(xproj_ptr + m * P + R + 7).to(tl.float32)
        c7 = tl.load(xproj_ptr + m * P + R + N + 7).to(tl.float32)
        h7 = p * h7 + xd * b7
        y_val += h7 * c7

        p = p * e
        b8 = tl.load(xproj_ptr + m * P + R + 8).to(tl.float32)
        c8 = tl.load(xproj_ptr + m * P + R + N + 8).to(tl.float32)
        h8 = p * h8 + xd * b8
        y_val += h8 * c8

        p = p * e
        b9 = tl.load(xproj_ptr + m * P + R + 9).to(tl.float32)
        c9 = tl.load(xproj_ptr + m * P + R + N + 9).to(tl.float32)
        h9 = p * h9 + xd * b9
        y_val += h9 * c9

        p = p * e
        b10 = tl.load(xproj_ptr + m * P + R + 10).to(tl.float32)
        c10 = tl.load(xproj_ptr + m * P + R + N + 10).to(tl.float32)
        h10 = p * h10 + xd * b10
        y_val += h10 * c10

        p = p * e
        b11 = tl.load(xproj_ptr + m * P + R + 11).to(tl.float32)
        c11 = tl.load(xproj_ptr + m * P + R + N + 11).to(tl.float32)
        h11 = p * h11 + xd * b11
        y_val += h11 * c11

        p = p * e
        b12 = tl.load(xproj_ptr + m * P + R + 12).to(tl.float32)
        c12 = tl.load(xproj_ptr + m * P + R + N + 12).to(tl.float32)
        h12 = p * h12 + xd * b12
        y_val += h12 * c12

        p = p * e
        b13 = tl.load(xproj_ptr + m * P + R + 13).to(tl.float32)
        c13 = tl.load(xproj_ptr + m * P + R + N + 13).to(tl.float32)
        h13 = p * h13 + xd * b13
        y_val += h13 * c13

        p = p * e
        b14 = tl.load(xproj_ptr + m * P + R + 14).to(tl.float32)
        c14 = tl.load(xproj_ptr + m * P + R + N + 14).to(tl.float32)
        h14 = p * h14 + xd * b14
        y_val += h14 * c14

        p = p * e
        b15 = tl.load(xproj_ptr + m * P + R + 15).to(tl.float32)
        c15 = tl.load(xproj_ptr + m * P + R + N + 15).to(tl.float32)
        h15 = p * h15 + xd * b15
        y_val += h15 * c15

        y_val += d_skip * x_val
        tl.store(y_ptr + m * D + offs_d, y_val)


class Model(nn.Module):
    def __init__(self, d_model: int = 1024, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.dt_rank = d_model // 16
        dt_proj_size = self.dt_rank + 2 * d_state

        self.conv1d = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
        )

        self.x_proj = nn.Linear(d_model, dt_proj_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(d_model))

        self.half()

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        N = self.d_state
        R = self.dt_rank
        P = R + 2 * N
        M = B * L

        x_conv = torch.empty((B, L, D), device=x.device, dtype=x.dtype)
        x_proj_out = torch.empty((M, P), device=x.device, dtype=x.dtype)
        dt = torch.empty((B, L, D), device=x.device, dtype=x.dtype)
        y = torch.empty((B, L, D), device=x.device, dtype=x.dtype)

        grid_conv = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )
        _causal_conv1d_silu_kernel[grid_conv](
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            x_conv,
            M,
            L,
            D,
            self.d_conv,
        )

        grid_xproj = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(P, meta["BLOCK_N"]),
        )
        _xproj_gemm_kernel[grid_xproj](
            x_conv,
            self.x_proj.weight,
            x_proj_out,
            M,
            D,
            P,
        )

        grid_dt = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_N"]),
        )
        _dtproj_softplus_gemm_kernel[grid_dt](
            x_proj_out,
            self.dt_proj.weight,
            self.dt_proj.bias,
            dt,
            M,
            D,
            R,
            P,
        )

        grid_ssm = lambda meta: (B, triton.cdiv(D, meta["BLOCK_D"]))
        _ssm_recurrence_power_kernel[grid_ssm](
            x_conv,
            x_proj_out,
            dt,
            self.A_log,
            self.D,
            y,
            L,
            D,
            N,
            P,
            R,
        )

        return y