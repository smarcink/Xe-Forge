import torch
import math
from torch import nn, Tensor
import triton
import triton.language as tl


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def precompute_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


@triton.jit
def fused_rmsnorm_rope_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    B, S, D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, HALF: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid indexes (b*S + s)
    s = pid % S

    row_off = pid * D
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = x * rstd * w  # fp32

    # Now apply RoPE per head. offs in [0, D). head_idx = offs // HD, lane = offs % HD.
    lane = offs % HD
    is_first_half = lane < HALF
    # Index into cos/sin: position s, frequency idx = lane if first half else lane - HALF
    freq_idx = tl.where(is_first_half, lane, lane - HALF)
    cs_off = s * HALF + freq_idx
    cos = tl.load(cos_ptr + cs_off, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off, mask=mask, other=0.0).to(tl.float32)

    # We need the "partner" value:
    # For first half (lane in [0,HALF)): x1=normed[lane], x2=normed[lane+HALF]
    #   out[lane] = x1*cos - x2*sin
    # For second half (lane in [HALF, HD)): x1=normed[lane-HALF], x2=normed[lane]
    #   out[lane] = x1*sin + x2*cos
    # So partner offset within head: if first half, partner = lane + HALF; else partner = lane - HALF.
    partner_lane = tl.where(is_first_half, lane + HALF, lane - HALF)
    head_base = (offs // HD) * HD
    partner_offs = head_base + partner_lane
    # gather partner from normed (which is the BLOCK_D vector)
    # use tl.load from a temp? We only have it in registers. We can recompute by gather via shared via tl.load on x and re-normalize? Better: store normed to shared via tl.store on a temp pointer — not available. Instead, do it via tl.where with shuffle pattern.
    # Trick: load x at partner positions from global and re-apply rstd*w.
    x_p = tl.load(x_ptr + row_off + partner_offs, mask=mask, other=0.0).to(tl.float32)
    w_p = tl.load(w_ptr + partner_offs, mask=mask, other=0.0).to(tl.float32)
    normed_p = x_p * rstd * w_p

    # x1, x2 selection
    x1 = tl.where(is_first_half, normed, normed_p)
    x2 = tl.where(is_first_half, normed_p, normed)

    out = tl.where(is_first_half, x1 * cos - x2 * sin, x1 * sin + x2 * cos)

    tl.store(out_ptr + row_off + offs, out.to(tl.float16), mask=mask)


class Model(nn.Module):
    def __init__(self, dim=4096, num_heads=32, seq_len=2048, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

        cos, sin = precompute_freqs(self.head_dim, seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.half()

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        assert D == self.dim
        out = torch.empty_like(x)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * S,)

        # Ensure cos/sin are fp32 for precision (they were converted to fp16 by .half())
        cos = self.rope_cos
        sin = self.rope_sin

        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, cos, sin, out,
            B, S, D, self.num_heads, self.head_dim, self.head_dim // 2,
            self.eps,
            BLOCK_D=BLOCK_D,
            num_warps=8,
            num_stages=2,
        )
        return out