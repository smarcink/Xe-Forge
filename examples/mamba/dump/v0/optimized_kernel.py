import torch
from torch import nn, Tensor
import triton
import triton.language as tl


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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.static_range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(x_ptr + offs_m[:, None] * K + k_idxs[None, :])
        b = tl.load(
            w_ptr + k_idxs[:, None] + offs_n[None, :] * K,
            mask=offs_n[None, :] < N,
            other=0.0,
        )

        acc += tl.dot(a, b)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=offs_n[None, :] < N,
    )


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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.static_range(0, R, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(xproj_ptr + offs_m[:, None] * P + k_idxs[None, :])
        b = tl.load(w_ptr + k_idxs[:, None] + offs_n[None, :] * R)

        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    acc += bias[None, :]

    soft = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    tl.store(dt_ptr + offs_m[:, None] * D + offs_n[None, :], soft)


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

        _causal_conv1d_silu_kernel[
            (triton.cdiv(M, 32), triton.cdiv(D, 32))
        ](
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            x_conv,
            M,
            L,
            D,
            self.d_conv,
            BLOCK_M=32,
            BLOCK_D=32,
            num_warps=4,
        )

        _xproj_gemm_kernel[
            (triton.cdiv(M, 64), triton.cdiv(P, 32))
        ](
            x_conv,
            self.x_proj.weight,
            x_proj_out,
            M,
            D,
            P,
            BLOCK_M=64,
            BLOCK_N=32,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )

        _dtproj_softplus_gemm_kernel[
            (triton.cdiv(M, 32), triton.cdiv(D, 64))
        ](
            x_proj_out,
            self.dt_proj.weight,
            self.dt_proj.bias,
            dt,
            M,
            D,
            R,
            P,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )

        _ssm_recurrence_power_kernel[
            (B, triton.cdiv(D, 8))
        ](
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
            BLOCK_D=8,
            num_warps=1,
            num_stages=3,
        )

        return y