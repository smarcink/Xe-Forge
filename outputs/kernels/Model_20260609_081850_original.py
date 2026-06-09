# Original: Model

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['D', 'I'],
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < m_count
    n_mask = offs_n < I

    x_ptrs = X_ptr + (start + offs_m)[:, None] * stride_xm + offs_k[None, :] * stride_xd
    w1_ptrs = W1_ptr + pid_e * stride_w1e + offs_k[:, None] * stride_w1d + offs_n[None, :] * stride_w1i
    w3_ptrs = W3_ptr + pid_e * stride_w3e + offs_k[:, None] * stride_w3d + offs_n[None, :] * stride_w3i

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, D, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w1 = tl.load(w1_ptrs, mask=n_mask[None, :], other=0.0)
        w3 = tl.load(w3_ptrs, mask=n_mask[None, :], other=0.0)
        acc1 += tl.dot(x, w1)
        acc3 += tl.dot(x, w3)
        x_ptrs += BLOCK_K * stride_xd
        w1_ptrs += BLOCK_K * stride_w1d
        w3_ptrs += BLOCK_K * stride_w3d

    sig = 1.0 / (1.0 + tl.exp(-acc1))
    out = (acc1 * sig) * acc3
    out = out.to(tl.float16)

    out_ptrs = OUT_ptr + (start + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_oi
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['D', 'I'],
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < m_count
    n_mask = offs_n < D

    inter_ptrs = INTER_ptr + (start + offs_m)[:, None] * stride_im + offs_k[None, :] * stride_ii
    w2_ptrs = W2_ptr + pid_e * stride_w2e + offs_k[:, None] * stride_w2i + offs_n[None, :] * stride_w2d

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, I, BLOCK_K):
        a = tl.load(inter_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)
        acc += tl.dot(a, w)
        inter_ptrs += BLOCK_K * stride_ii
        w2_ptrs += BLOCK_K * stride_w2i

    w_vals = tl.load(sorted_weights + start + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    out = acc * w_vals[:, None]
    out = out.to(tl.float16)

    tok_ids = tl.load(sorted_token_ids + start + offs_m, mask=m_mask, other=0)
    out_ptrs = OUT_ptr + tok_ids[:, None] * stride_om + offs_n[None, :] * stride_od

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

    def forward(self, hidden_states: Tensor) -> Tensor:
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

        max_tiles_1 = triton.cdiv(M, 32) * triton.cdiv(I, 128)
        grid1 = (E, max_tiles_1)
        fused_gate_up_silu_kernel[grid1](
            permuted_tokens, self.w1, self.w3, intermediate,
            expert_offsets,
            M, D, I,
            permuted_tokens.stride(0), permuted_tokens.stride(1),
            self.w1.stride(0), self.w1.stride(1), self.w1.stride(2),
            self.w3.stride(0), self.w3.stride(1), self.w3.stride(2),
            intermediate.stride(0), intermediate.stride(1),
        )

        output = torch.zeros(N, D, device=flat.device, dtype=torch.float16)

        max_tiles_2 = triton.cdiv(M, 32) * triton.cdiv(D, 128)
        grid2 = (E, max_tiles_2)
        fused_down_scatter_kernel[grid2](
            intermediate, self.w2, output,
            sorted_token_ids, sorted_weights, expert_offsets,
            M, D, I,
            intermediate.stride(0), intermediate.stride(1),
            self.w2.stride(0), self.w2.stride(1), self.w2.stride(2),
            output.stride(0), output.stride(1),
        )

        return output.view(B, S, D)