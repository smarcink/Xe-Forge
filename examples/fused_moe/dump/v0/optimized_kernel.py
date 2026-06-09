import torch
import torch.nn.functional as F
from torch import nn, Tensor
import triton
import triton.language as tl


def _gate_up_autotune_configs():
    # NOTE: grf_mode is intentionally NOT in triton.Config; it is passed at launch.
    # For the target shapes D=1024 and I=2816, all listed BN/BK divide N/K
    # dimensions, so EVEN_N/EVEN_K allow the compiler to remove RHS/K boundary
    # checks while still keeping routed-row M boundary checks.
    return [
        # Original strong configs, specialized for even N/K.
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=4,  num_stages=4),

        # Smaller/occupancy-friendly tiles for the two-accumulator gate/up kernel.
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 2,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 2,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),

        # Wider N tiles reduce programs over I=2816.
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 2,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=3),

        # Per-expert routed rows are roughly M/E=256 for the target, so BM=128
        # often balances occupancy and arithmetic intensity.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),

        # Required large XPU tile. Gate/up has two FP32 accumulators, so keep only
        # the most relevant large-tile candidate to avoid excessive tuning cost.
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 16, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=32, num_stages=3),
    ]


def _down_autotune_configs():
    # Down projection has only one accumulator, so it can profit more from larger
    # XPU DPAS tiles and high warp counts. grf_mode is passed as a launch option.
    return [
        # Conservative fallbacks.
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 2,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 2,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=4,  num_stages=3),

        # Wider output columns.
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 2,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=3),

        # Strong medium/large DPAS tiles.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=16, num_stages=2),

        # Large XPU-oriented candidates, including the requested 256x256 tile.
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 16, 'GROUP_M': 1,
                       'EVEN_N': True, 'EVEN_K': True}, num_warps=32, num_stages=3),
    ]


@triton.autotune(
    configs=_gate_up_autotune_configs(),
    key=['M', 'D', 'I'],
)
@triton.jit
def fused_gate_up_silu_kernel(
    X_ptr,
    W1_ptr,
    W3_ptr,
    OUT_ptr,
    expert_offsets,
    M, D, I,
    stride_xm, stride_xd,
    stride_w1e, stride_w1i, stride_w1d,
    stride_w3e, stride_w3i, stride_w3d,
    stride_om, stride_oi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid = tl.program_id(1)

    start = tl.load(expert_offsets + pid_e)
    end = tl.load(expert_offsets + pid_e + 1)
    m_count = end - start
    if m_count <= 0:
        return

    num_pid_m = tl.cdiv(m_count, BLOCK_M)
    num_pid_n = tl.cdiv(I, BLOCK_N)
    total = num_pid_m * num_pid_n
    if pid >= total:
        return

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    row0 = start + pid_m * BLOCK_M
    col0 = tl.multiple_of(pid_n * BLOCK_N, BLOCK_N)

    x_bp = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, D),
        strides=(stride_xm, stride_xd),
        offsets=(row0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w1_bp = tl.make_block_ptr(
        base=W1_ptr + pid_e * stride_w1e,
        shape=(D, I),
        strides=(stride_w1d, stride_w1i),
        offsets=(0, col0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    w3_bp = tl.make_block_ptr(
        base=W3_ptr + pid_e * stride_w3e,
        shape=(D, I),
        strides=(stride_w3d, stride_w3i),
        offsets=(0, col0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, D, BLOCK_K):
        # Routed rows are generally not aligned to BLOCK_M, so retain M boundary.
        if EVEN_K:
            x = tl.load(x_bp, boundary_check=(0,), padding_option="zero")
        else:
            x = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")

        if EVEN_N and EVEN_K:
            w1 = tl.load(w1_bp)
            w3 = tl.load(w3_bp)
        else:
            w1 = tl.load(w1_bp, boundary_check=(0, 1), padding_option="zero")
            w3 = tl.load(w3_bp, boundary_check=(0, 1), padding_option="zero")

        acc1 += tl.dot(x, w1)
        acc3 += tl.dot(x, w3)

        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w1_bp = tl.advance(w1_bp, (BLOCK_K, 0))
        w3_bp = tl.advance(w3_bp, (BLOCK_K, 0))

    sig = 1.0 / (1.0 + tl.math.exp2(-acc1 * 1.4426950408889634))
    out = (acc1 * sig * acc3).to(tl.float16)

    out_bp = tl.make_block_ptr(
        base=OUT_ptr,
        shape=(M, I),
        strides=(stride_om, stride_oi),
        offsets=(row0, col0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    if EVEN_N:
        tl.store(out_bp, out, boundary_check=(0,))
    else:
        tl.store(out_bp, out, boundary_check=(0, 1))


@triton.autotune(
    configs=_down_autotune_configs(),
    key=['M', 'D', 'I'],
)
@triton.jit
def fused_down_scatter_kernel(
    INTER_ptr,
    W2_ptr,
    OUT_ptr,
    sorted_token_ids,
    sorted_weights,
    expert_offsets,
    M, D, I,
    stride_im, stride_ii,
    stride_w2e, stride_w2d, stride_w2i,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid = tl.program_id(1)

    start = tl.load(expert_offsets + pid_e)
    end = tl.load(expert_offsets + pid_e + 1)
    m_count = end - start
    if m_count <= 0:
        return

    num_pid_m = tl.cdiv(m_count, BLOCK_M)
    num_pid_n = tl.cdiv(D, BLOCK_N)
    total = num_pid_m * num_pid_n
    if pid >= total:
        return

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    row0 = start + pid_m * BLOCK_M
    col0 = tl.multiple_of(pid_n * BLOCK_N, BLOCK_N)

    inter_bp = tl.make_block_ptr(
        base=INTER_ptr,
        shape=(M, I),
        strides=(stride_im, stride_ii),
        offsets=(row0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w2_bp = tl.make_block_ptr(
        base=W2_ptr + pid_e * stride_w2e,
        shape=(I, D),
        strides=(stride_w2i, stride_w2d),
        offsets=(0, col0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, I, BLOCK_K):
        if EVEN_K:
            a = tl.load(inter_bp, boundary_check=(0,), padding_option="zero")
        else:
            a = tl.load(inter_bp, boundary_check=(0, 1), padding_option="zero")

        if EVEN_N and EVEN_K:
            w = tl.load(w2_bp)
        else:
            w = tl.load(w2_bp, boundary_check=(0, 1), padding_option="zero")

        acc += tl.dot(a, w)

        inter_bp = tl.advance(inter_bp, (0, BLOCK_K))
        w2_bp = tl.advance(w2_bp, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = col0 + tl.arange(0, BLOCK_N)
    m_mask = offs_m < m_count

    w_vals = tl.load(sorted_weights + start + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    out = (acc * w_vals[:, None]).to(tl.float16)

    tok_ids = tl.load(sorted_token_ids + start + offs_m, mask=m_mask, other=0)
    out_ptrs = OUT_ptr + tok_ids[:, None] * stride_om + offs_n[None, :] * stride_od

    if EVEN_N:
        tl.atomic_add(out_ptrs, out, mask=m_mask[:, None])
    else:
        n_mask = offs_n < D
        tl.atomic_add(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        intermediate_size: int = 2816,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts, bias=False)

        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))

        nn.init.kaiming_uniform_(self.w1, a=2.236)
        nn.init.kaiming_uniform_(self.w2, a=2.236)
        nn.init.kaiming_uniform_(self.w3, a=2.236)

        self.half()

        self._w1_packed = None
        self._w2_packed = None
        self._w3_packed = None
        self._packed_versions = None

    def _ensure_xpu_fp16(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.device.type != "xpu":
            hidden_states = hidden_states.to("xpu", dtype=torch.float16)
        elif hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(dtype=torch.float16)

        moved_or_cast = False
        if self.w1.device.type != "xpu" or self.gate.weight.device.type != "xpu":
            self.to("xpu")
            moved_or_cast = True
        if self.w1.dtype != torch.float16 or self.gate.weight.dtype != torch.float16:
            self.half()
            moved_or_cast = True
        if moved_or_cast:
            self._w1_packed = None
            self._w2_packed = None
            self._w3_packed = None
            self._packed_versions = None
        return hidden_states

    def _ensure_packed_weights(self):
        versions = (self.w1._version, self.w2._version, self.w3._version)
        if (
            self._w1_packed is None
            or self._w2_packed is None
            or self._w3_packed is None
            or self._packed_versions != versions
            or self._w1_packed.device != self.w1.device
            or self._w2_packed.device != self.w2.device
            or self._w3_packed.device != self.w3.device
        ):
            # Pack once into Triton's logical RHS layouts:
            #   W1/W3 original [E, I, D] -> packed [E, D, I]
            #   W2    original [E, D, I] -> packed [E, I, D]
            self._w1_packed = self.w1.detach().permute(0, 2, 1).contiguous()
            self._w3_packed = self.w3.detach().permute(0, 2, 1).contiguous()
            self._w2_packed = self.w2.detach().permute(0, 2, 1).contiguous()
            self._packed_versions = versions
        return self._w1_packed, self._w2_packed, self._w3_packed

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self._ensure_xpu_fp16(hidden_states)
        w1_packed, w2_packed, w3_packed = self._ensure_packed_weights()

        B, S, D = hidden_states.shape
        N = B * S
        E = self.num_experts
        I = self.intermediate_size
        K = self.top_k

        flat = hidden_states.view(N, D)

        scores = self.gate(flat).float()
        probs = F.softmax(scores, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)

        token_ids = torch.arange(N, device=flat.device).unsqueeze(1).expand(-1, K).reshape(-1)

        sorted_order = flat_indices.argsort(stable=True)
        sorted_token_ids = token_ids[sorted_order].to(torch.int32)
        sorted_expert_ids = flat_indices[sorted_order]
        sorted_weights = flat_weights[sorted_order].contiguous()

        M = N * K

        permuted_tokens = flat[sorted_token_ids.long()].contiguous()

        expert_offsets = torch.zeros(E + 1, device=flat.device, dtype=torch.int32)
        counts = torch.bincount(sorted_expert_ids, minlength=E)
        expert_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

        intermediate = torch.empty(M, I, device=flat.device, dtype=torch.float16)

        grid1 = lambda META: (
            E,
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(I, META["BLOCK_N"]),
        )
        fused_gate_up_silu_kernel[grid1](
            permuted_tokens, w1_packed, w3_packed, intermediate,
            expert_offsets,
            M, D, I,
            permuted_tokens.stride(0), permuted_tokens.stride(1),
            w1_packed.stride(0), w1_packed.stride(2), w1_packed.stride(1),
            w3_packed.stride(0), w3_packed.stride(2), w3_packed.stride(1),
            intermediate.stride(0), intermediate.stride(1),
            grf_mode="256",
        )

        output = torch.zeros(N, D, device=flat.device, dtype=torch.float16)

        grid2 = lambda META: (
            E,
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(D, META["BLOCK_N"]),
        )
        fused_down_scatter_kernel[grid2](
            intermediate, w2_packed, output,
            sorted_token_ids, sorted_weights, expert_offsets,
            M, D, I,
            intermediate.stride(0), intermediate.stride(1),
            w2_packed.stride(0), w2_packed.stride(2), w2_packed.stride(1),
            output.stride(0), output.stride(1),
            grf_mode="256",
        )

        return output.view(B, S, D)