import torch
from torch import nn, Tensor

def _make_relative_position_bias(window_size, num_heads, dtype):
    Wh, Ww = window_size
    table = nn.init.trunc_normal_(
        torch.empty((2 * Wh - 1) * (2 * Ww - 1), num_heads, dtype=dtype),
        std=0.02,
    )
    coords = torch.stack(torch.meshgrid(
        torch.arange(Wh), torch.arange(Ww), indexing="ij"))            # 2, Wh, Ww
    coords = coords.flatten(1)                                          # 2, N
    rel = coords[:, :, None] - coords[:, None, :]                       # 2, N, N
    rel = rel.permute(1, 2, 0).contiguous()                             # N, N, 2
    rel[..., 0] += Wh - 1
    rel[..., 1] += Ww - 1
    rel[..., 0] *= 2 * Ww - 1
    idx = rel.sum(-1).flatten()                                         # N*N
    bias = table[idx].view(Wh * Ww, Wh * Ww, num_heads).permute(2, 0, 1)  # H, N, N
    return bias.unsqueeze(0).contiguous()                               # 1, H, N, N


class Model(nn.Module):
    """Simplified Swin block: LN -> W-MSA -> residual -> LN -> MLP -> residual."""

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **_):
        super().__init__()
        assert num_heads == 1, "simplified path: single-head only"
        assert tuple(shift_size) == (0, 0), "simplified path: no shift"
        hidden = int(dim * mlp_ratio)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = list(window_size)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=True)

        # Match reference initialization for fc1/fc2 so weight-copy comparisons
        # against the original swin_block_fp16.py are numerically meaningful.
        for m in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

        # Pre-gather relative-position bias once: (1, num_heads, N, N).
        rpe = _make_relative_position_bias(self.window_size, num_heads, dtype=torch.float32)
        self.register_buffer("rpe_bias", rpe)

        # Run the whole block in fp16. nn.LayerNorm CUDA kernels still
        # accumulate mean/var in fp32 internally for fp16 inputs.
        self.half()

    def _attn(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        # window partition: (B, H, W, C) -> (B*nH*nW, N, C)
        x = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, N, C)

        qkv = self.qkv(x).reshape(-1, N, 3, C).permute(2, 0, 1, 3)        # 3, Bw, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1) + self.rpe_bias          # Bw, N, N
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(-1, N, C)
        x = self.proj(x)

        # window reverse: (B*nH*nW, N, C) -> (B, H, W, C)
        x = x.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attn(self.norm1(x))
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x


def get_inputs():
    return [torch.randn(1, 720, 1280, 32, dtype=torch.float16)]


def get_init_inputs():
    return [32, 1, [4, 4], [0, 0], 4]
