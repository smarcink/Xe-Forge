import torch
from torch import nn, Tensor
import triton
import triton.language as tl


def _make_relative_position_bias(window_size, num_heads, dtype):
    Wh, Ww = window_size
    table = nn.init.trunc_normal_(
        torch.empty((2 * Wh - 1) * (2 * Ww - 1), num_heads, dtype=dtype),
        std=0.02,
    )
    coords = torch.stack(torch.meshgrid(
        torch.arange(Wh), torch.arange(Ww), indexing="ij"))
    coords = coords.flatten(1)
    rel = coords[:, :, None] - coords[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[..., 0] += Wh - 1
    rel[..., 1] += Ww - 1
    rel[..., 0] *= 2 * Ww - 1
    idx = rel.sum(-1).flatten()
    bias = table[idx].view(Wh * Ww, Wh * Ww, num_heads).permute(2, 0, 1)
    return bias.unsqueeze(0).contiguous()


def _swin_autotune_configs():
    configs = []
    # With phase-local dense-weight loads, larger WPP no longer amortizes the
    # same long-lived weight tiles; smaller WPP avoids excessive unrolled code
    # and register pressure while still allowing autotune to pick mild reuse.
    for wpp in [1, 2]:
        for nw in [1, 2, 4, 8]:
            for ns in [2, 3]:
                configs.append(
                    triton.Config({"WINDOWS_PER_PROG": wpp}, num_warps=nw, num_stages=ns)
                )
    return configs


@triton.autotune(configs=_swin_autotune_configs(), key=["B", "H", "W", "nH", "nW"])
@triton.jit
def swin_block_kernel(
    X_ptr, Y_ptr,
    ln1_w_ptr, ln1_b_ptr,
    q_w_t_ptr, q_b_ptr,
    k_w_t_ptr, k_b_ptr,
    v_w_t_ptr, v_b_ptr,
    proj_w_t_ptr, proj_b_ptr,
    rpe_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_t_ptr, fc1_b_ptr,
    fc2_w_t_ptr, fc2_b_ptr,
    B, H, W,
    nH, nW,
    C: tl.constexpr,
    HIDDEN: tl.constexpr,
    N: tl.constexpr,
    WS: tl.constexpr,
    WINDOWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    total_windows = B * nH * nW
    base_w = pid * WINDOWS_PER_PROG

    n_idx = tl.arange(0, N)
    c_idx = tl.arange(0, C)
    h_idx = tl.arange(0, HIDDEN)

    scale = 1.0 / tl.sqrt(tl.full((), C, tl.float32))

    for wi in tl.static_range(0, WINDOWS_PER_PROG):
        win_id = base_w + wi
        valid_w = win_id < total_windows
        safe_win = tl.where(valid_w, win_id, 0)

        b = safe_win // (nH * nW)
        rem = safe_win % (nH * nW)
        ih = rem // nW
        iw = rem % nW

        i_in = n_idx // WS
        j_in = n_idx % WS
        row = ih * WS + i_in
        col = iw * WS + j_in

        x_off = (
            b * (H * W * C)
            + row[:, None] * (W * C)
            + col[:, None] * C
            + c_idx[None, :]
        )

        x = tl.load(X_ptr + x_off, mask=valid_w, other=0.0)
        x_f = x.to(tl.float32)
        residual1 = x_f

        # LN1 parameters are small and loaded only for the LN1 phase.
        ln1_w = tl.load(ln1_w_ptr + c_idx).to(tl.float32)
        ln1_b = tl.load(ln1_b_ptr + c_idx).to(tl.float32)

        mean = tl.sum(x_f, axis=1) / C
        xc = x_f - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / C
        inv = 1.0 / tl.sqrt(var + 1.0e-5)
        ln1_h = (xc * inv[:, None] * ln1_w[None, :] + ln1_b[None, :]).to(tl.float16)

        # Q/K/V weights are pre-transposed and loaded one at a time, directly
        # before use. This shortens lifetimes versus keeping q_w/k_w/v_w and
        # their transposes live across the whole fused block.
        q_w_t = tl.load(q_w_t_ptr + c_idx[:, None] * C + c_idx[None, :])
        q_b = tl.load(q_b_ptr + c_idx).to(tl.float32)
        q_h = ((tl.dot(ln1_h, q_w_t).to(tl.float32) + q_b[None, :]) * scale).to(tl.float16)

        k_w_t = tl.load(k_w_t_ptr + c_idx[:, None] * C + c_idx[None, :])
        k_b = tl.load(k_b_ptr + c_idx).to(tl.float32)
        k_h = (tl.dot(ln1_h, k_w_t).to(tl.float32) + k_b[None, :]).to(tl.float16)

        v_w_t = tl.load(v_w_t_ptr + c_idx[:, None] * C + c_idx[None, :])
        v_b = tl.load(v_b_ptr + c_idx).to(tl.float32)
        v_h = (tl.dot(ln1_h, v_w_t).to(tl.float32) + v_b[None, :]).to(tl.float16)

        attn = tl.dot(q_h, tl.trans(k_h)).to(tl.float32)
        rpe = tl.load(rpe_ptr + n_idx[:, None] * N + n_idx[None, :]).to(tl.float32)
        attn = attn + rpe

        amax = tl.max(attn, axis=1)
        attn = attn - amax[:, None]
        attn_e = tl.exp(attn)
        asum = tl.sum(attn_e, axis=1)
        attn_p = (attn_e / asum[:, None]).to(tl.float16)

        out_h = tl.dot(attn_p, v_h).to(tl.float16)

        # Projection weights are not live during QKV/softmax.
        proj_w_t = tl.load(proj_w_t_ptr + c_idx[:, None] * C + c_idx[None, :])
        proj_b = tl.load(proj_b_ptr + c_idx).to(tl.float32)
        proj_out = tl.dot(out_h, proj_w_t).to(tl.float32) + proj_b[None, :]
        x1 = proj_out + residual1
        residual2 = x1

        # LN2 parameters are loaded only for the LN2 phase.
        ln2_w = tl.load(ln2_w_ptr + c_idx).to(tl.float32)
        ln2_b = tl.load(ln2_b_ptr + c_idx).to(tl.float32)

        mean2 = tl.sum(x1, axis=1) / C
        xc2 = x1 - mean2[:, None]
        var2 = tl.sum(xc2 * xc2, axis=1) / C
        inv2 = 1.0 / tl.sqrt(var2 + 1.0e-5)
        ln2_h = (xc2 * inv2[:, None] * ln2_w[None, :] + ln2_b[None, :]).to(tl.float16)

        # MLP weights are phase-local: FC1 is loaded after LN2; FC2 only after
        # GELU. This reduces GRF pressure and spill risk substantially.
        fc1_w_t = tl.load(fc1_w_t_ptr + c_idx[:, None] * HIDDEN + h_idx[None, :])
        fc1_b = tl.load(fc1_b_ptr + h_idx).to(tl.float32)
        h_act = tl.dot(ln2_h, fc1_w_t).to(tl.float32) + fc1_b[None, :]
        h_gelu = 0.5 * h_act * (1.0 + tl.erf(h_act * 0.7071067811865475))
        h_gelu_h = h_gelu.to(tl.float16)

        fc2_w_t = tl.load(fc2_w_t_ptr + h_idx[:, None] * C + c_idx[None, :])
        fc2_b = tl.load(fc2_b_ptr + c_idx).to(tl.float32)
        fc2_out = tl.dot(h_gelu_h, fc2_w_t).to(tl.float32) + fc2_b[None, :]
        y_h = (fc2_out + residual2).to(tl.float16)

        # Recompute the output offsets so the full [N, C] offset tile does not
        # need to remain live throughout all fused phases.
        row_s = ih * WS + (n_idx // WS)
        col_s = iw * WS + (n_idx % WS)
        y_off = (
            b * (H * W * C)
            + row_s[:, None] * (W * C)
            + col_s[:, None] * C
            + c_idx[None, :]
        )
        tl.store(Y_ptr + y_off, y_h, mask=valid_w)


class Model(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **_):
        super().__init__()
        assert num_heads == 1
        assert int(shift_size) == 0

        hidden = int(dim * mlp_ratio)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = [int(window_size), int(window_size)]
        self.hidden = hidden

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=True)

        for m in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

        rpe = _make_relative_position_bias(self.window_size, num_heads, dtype=torch.float32)
        self.register_buffer("rpe_bias", rpe)

        self.half()

        self._cached_weights = None
        self._cached_device = None

    def _ensure_xpu(self):
        # Safe for harnesses that do not explicitly call model.to("xpu").
        try:
            p = next(self.parameters())
            if p.device.type != "xpu":
                self.to("xpu")
                self._cached_weights = None
                self._cached_device = None
        except StopIteration:
            pass

    def _get_packed_weights(self):
        C = self.dim
        dev = self.qkv.weight.device
        if self._cached_weights is None or self._cached_device != dev:
            qkv_w = self.qkv.weight
            qkv_b = self.qkv.bias

            # Pre-transpose once on XPU so the kernel can load the exact layout
            # consumed by tl.dot without keeping original tiles + tl.trans live.
            q_w_t = qkv_w[0:C].t().contiguous()
            k_w_t = qkv_w[C:2 * C].t().contiguous()
            v_w_t = qkv_w[2 * C:3 * C].t().contiguous()

            q_b = qkv_b[0:C].contiguous()
            k_b = qkv_b[C:2 * C].contiguous()
            v_b = qkv_b[2 * C:3 * C].contiguous()

            proj_w_t = self.proj.weight.t().contiguous()
            fc1_w_t = self.fc1.weight.t().contiguous()
            fc2_w_t = self.fc2.weight.t().contiguous()

            self._cached_weights = (
                q_w_t, q_b,
                k_w_t, k_b,
                v_w_t, v_b,
                proj_w_t,
                fc1_w_t,
                fc2_w_t,
            )
            self._cached_device = dev
        return self._cached_weights

    def forward(self, x: Tensor) -> Tensor:
        if x.device.type != "xpu":
            x = x.to("xpu", dtype=torch.float16)
        elif x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)

        self._ensure_xpu()

        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        assert Wh == Ww == 4
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        x = x.contiguous()
        y = torch.empty_like(x)

        (
            q_w_t, q_b,
            k_w_t, k_b,
            v_w_t, v_b,
            proj_w_t,
            fc1_w_t,
            fc2_w_t,
        ) = self._get_packed_weights()

        total_windows = B * nH * nW
        grid = lambda meta: (triton.cdiv(total_windows, meta["WINDOWS_PER_PROG"]),)

        swin_block_kernel[grid](
            x, y,
            self.norm1.weight, self.norm1.bias,
            q_w_t, q_b,
            k_w_t, k_b,
            v_w_t, v_b,
            proj_w_t, self.proj.bias,
            self.rpe_bias,
            self.norm2.weight, self.norm2.bias,
            fc1_w_t, self.fc1.bias,
            fc2_w_t, self.fc2.bias,
            B, H, W,
            nH, nW,
            C,
            self.hidden,
            N, Wh,
        )
        return y


def get_init_inputs():
    return [32, 1, 4, 0, 4]


def get_inputs():
    return [torch.randn(1, 720, 1280, 32, dtype=torch.float16)]