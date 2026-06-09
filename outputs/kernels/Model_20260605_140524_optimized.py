# Optimized: Model
# Speedup: 11.22x
# Stages: ['block_pointers', 'device_specific']

from typing import Callable

import torch
import triton
import triton.language as tl


# Device-specific autotune space for the active non-causal inference path.
# Targets long-context D=128 attention on Intel XPU:
#   - BN=128 halves K/V loop iterations vs original BN=64
#   - 16/32 warp sweep lets backend choose best XMX occupancy
#   - BM=256 candidates are included conservatively with num_stages=1
#   - GROUP_SIZE_M is used by the flattened 1D launch/schedule
_noncausal_fwd_configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 4}, num_stages=3, num_warps=16),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 4}, num_stages=3, num_warps=32),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 4}, num_stages=2, num_warps=16),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 4}, num_stages=2, num_warps=32),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "GROUP_SIZE_M": 4}, num_stages=1, num_warps=16),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "GROUP_SIZE_M": 4}, num_stages=1, num_warps=32),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_SIZE_M": 4}, num_stages=1, num_warps=16),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_SIZE_M": 4}, num_stages=1, num_warps=32),
]


# Specialized non-causal forward path used by Model.forward.
# Uses descriptor-based 2D tile loads/stores and avoids materializing M.
@triton.autotune(configs=_noncausal_fwd_configs, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd_noncausal(
    sm_scale,
    Q,
    K,
    V,
    O,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    dtype = tl.float16

    # Flattened 1D head-major schedule.  For the target long-context shape,
    # original grid=(Z,H,M_tile) can schedule neighboring programs across heads.
    # This layout keeps adjacent programs on M tiles of the same head, improving
    # temporal reuse of the streamed K/V sequence in cache.
    num_pid_m: tl.constexpr = tl.cdiv(N_CTX, BLOCK_M)
    pid = tl.program_id(0)

    off_hz = pid // num_pid_m
    pid_m = pid - off_hz * num_pid_m

    # Grouped-M form; for this 1D attention grid it preserves head-major order
    # while making the grouping explicit and tunable.
    group_id_m = pid_m // GROUP_SIZE_M
    group_off_m = pid_m - group_id_m * GROUP_SIZE_M
    start_m = group_id_m * GROUP_SIZE_M + group_off_m

    off_z = off_hz // H
    off_h = off_hz - off_z * H

    y_dim: tl.constexpr = Z * H * N_CTX

    desc_q = tl.make_tensor_descriptor(
        Q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_k = tl.make_tensor_descriptor(
        K,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = tl.make_tensor_descriptor(
        V,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = tl.make_tensor_descriptor(
        O,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    q = desc_q.load([qo_offset_y, 0])

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.4426950408889634

    offsetkv_y = offset_y
    for start_n in tl.range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetkv_y, 0]).T
        qk = tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        acc = acc * alpha[:, None]

        v = desc_v.load([offsetkv_y, 0])
        acc = tl.dot(p.to(dtype), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetkv_y += BLOCK_N

    acc = acc / l_i[:, None]
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


# pylint: disable=unused-argument
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        acc = acc * alpha[:, None]

        v = desc_v.load([offsetv_y, 0])
        acc = tl.dot(p.to(dtype), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    Q,
    K,
    V,
    O,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    if N_CTX <= 512:
        start_m = tl.program_id(0)
        off_hz = tl.program_id(2)
        off_z = off_hz // H
        off_h = off_hz % H
    else:
        off_z = tl.program_id(0)
        off_h = tl.program_id(1)
        start_m = tl.program_id(2)

    y_dim = Z * H * N_CTX

    desc_q = tl.make_tensor_descriptor(
        Q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM]
    )
    desc_v = tl.make_tensor_descriptor(
        V, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM]
    )
    desc_k = tl.make_tensor_descriptor(
        K, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM]
    )
    desc_o = tl.make_tensor_descriptor(
        O, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM]
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.4426950408889634
    q = desc_q.load([qo_offset_y, 0])

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
        )

    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    if N_CTX <= 512:
        off_hz = tl.program_id(2)
    else:
        off_hz = tl.program_id(0) * H + tl.program_id(1)

    desc_m = tl.make_tensor_descriptor(
        base=M + off_hz * N_CTX,
        shape=[N_CTX],
        strides=[1],
        block_shape=[BLOCK_M],
    )
    desc_m.store([start_m * BLOCK_M], m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [128, 256]
    for BN in [32, 64]
    for s in [2, 3, 4]
    for w in [8, 16, 32]
]

tuner = triton.autotune(configs, key=["N_CTX", "HEAD_DIM", "STAGE"])


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_m0 = tl.program_id(0) * BLOCK_M

    o_bp = tl.make_block_ptr(
        base=O + off_hz * N_CTX * HEAD_DIM,
        shape=(N_CTX, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(off_m0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    do_bp = tl.make_block_ptr(
        base=DO + off_hz * N_CTX * HEAD_DIM,
        shape=(N_CTX, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(off_m0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    o = tl.load(o_bp, boundary_check=(0, 1), padding_option="zero")
    do = tl.load(do_bp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    offs_m = off_m0 + tl.arange(0, BLOCK_M)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta, mask=offs_m < N_CTX)


@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    k,
    v,
    sm_scale,
    DO,
    M,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    qT_desc = tl.make_tensor_descriptor(
        Q,
        shape=[HEAD_DIM, N_CTX],
        strides=[stride_d, stride_tok],
        block_shape=[HEAD_DIM, BLOCK_M1],
    )
    do_desc = tl.make_tensor_descriptor(
        DO,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_M1, HEAD_DIM],
    )

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

    curr_m = start_m
    step_m = BLOCK_M1

    for blk_idx in range(num_steps):
        qT = qT_desc.load([0, start_m + blk_idx * step_m])

        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)

        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])

        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        do = do_desc.load([start_m + blk_idx * step_m, 0])

        dv += tl.dot(pT.to(tl.float16), do)

        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = (pT * (dpT - Di[None, :])).to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))

        curr_m += step_m

    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,
    do,
    m,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    kT_desc = tl.make_tensor_descriptor(
        K,
        shape=[HEAD_DIM, N_CTX],
        strides=[stride_d, stride_tok],
        block_shape=[HEAD_DIM, BLOCK_N2],
    )
    vT_desc = tl.make_tensor_descriptor(
        V,
        shape=[HEAD_DIM, N_CTX],
        strides=[stride_d, stride_tok],
        block_shape=[HEAD_DIM, BLOCK_N2],
    )

    Di = tl.load(D + offs_m)

    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

    curr_n = start_n
    step_n = BLOCK_N2

    for blk_idx in range(num_steps):
        kT = kT_desc.load([0, start_n + blk_idx * step_n])
        vT = vT_desc.load([0, start_n + blk_idx * step_n])

        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)

        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(do, vT).to(tl.float32)
        ds = (p * (dp - Di[:, None])).to(tl.float16)
        dq += tl.dot(ds, tl.trans(kT))

        curr_n += step_n

    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    start_n = pid * BLOCK_N1
    start_m = start_n
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

    k_desc = tl.make_tensor_descriptor(
        K,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_N1, HEAD_DIM],
    )
    v_desc = tl.make_tensor_descriptor(
        V,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_N1, HEAD_DIM],
    )

    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        start_m,
        num_steps,
        MASK=True,
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        start_m,
        num_steps,
        MASK=False,
    )

    dv_desc = tl.make_tensor_descriptor(
        DV,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_N1, HEAD_DIM],
    )
    dk_desc = tl.make_tensor_descriptor(
        DK,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_N1, HEAD_DIM],
    )

    dv_desc.store([start_n, 0], dv)
    dk_desc.store([start_n, 0], dk * sm_scale)

    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q_desc = tl.make_tensor_descriptor(
        Q,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_M2, HEAD_DIM],
    )
    do_desc = tl.make_tensor_descriptor(
        DO,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_M2, HEAD_DIM],
    )

    q = q_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])

    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    m = tl.load(M + offs_m)[:, None]

    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,
        MASK=True,
    )

    end_n -= num_steps * MASK_BLOCK_N2
    num_steps = end_n // BLOCK_N2

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,
        MASK=False,
    )

    dq_desc = tl.make_tensor_descriptor(
        DQ,
        shape=[N_CTX, HEAD_DIM],
        strides=[stride_tok, stride_d],
        block_shape=[BLOCK_M2, HEAD_DIM],
    )
    dq_desc.store([start_m, 0], dq * LN2)


class _attention(torch.autograd.Function):
    tune_attn_fwd: Callable = None
    attn_fwd: Callable = None

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        o = torch.empty_like(q)
        stage = 3 if causal else 1

        if q.shape[2] <= 512:
            grid = lambda args: (
                triton.cdiv(q.shape[2], args["BLOCK_M"]),
                1,
                q.shape[0] * q.shape[1],
            )
        else:
            grid = lambda args: (
                q.shape[0],
                q.shape[1],
                triton.cdiv(q.shape[2], args["BLOCK_M"]),
            )

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attention.tune_attn_fwd[grid](
            sm_scale,
            M,
            q.shape[0],
            q.shape[1],
            q,
            k,
            v,
            o,
            N_CTX=q.shape[2],
            HEAD_DIM=Lk,
            STAGE=stage,
            grf_mode="auto",
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BATCH, N_HEAD, N_CTX = q.shape[:3]

        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 16, 3
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634

        arg_k = k * (ctx.sm_scale * RCP_LN2)
        delta = torch.empty_like(M)

        _attn_bwd_preprocess[(N_CTX // PRE_BLOCK, BATCH * N_HEAD)](
            o,
            do,
            delta,
            BATCH,
            N_HEAD,
            N_CTX,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        _attn_bwd[(N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            N_HEAD,
            N_CTX,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dq, dk, dv, None, None, None, None


def _ensure_xpu_fp16_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16).contiguous()
    if not x.is_contiguous():
        return x.contiguous()
    return x


def attn_fwd_launch(Q, K, V, causal: bool = False, sm_scale: float = 0.125):
    # Q,K,V: [Z,H,N_CTX,D]
    Q = _ensure_xpu_fp16_contiguous(Q)
    K = _ensure_xpu_fp16_contiguous(K)
    V = _ensure_xpu_fp16_contiguous(V)

    assert Q.shape == K.shape == V.shape
    Z, H, N_CTX, D = Q.shape

    O = torch.empty_like(Q)

    if not causal:
        # Required for the flattened/swizzled schedule: use a 1D grid.
        grid = lambda META: (Z * H * triton.cdiv(N_CTX, META["BLOCK_M"]),)

        _attn_fwd_noncausal[grid](
            sm_scale,
            Q,
            K,
            V,
            O,
            Z,
            H,
            N_CTX=N_CTX,
            HEAD_DIM=D,
            grf_mode="auto",
        )
        return O

    M = torch.empty((Z, H, N_CTX), device=Q.device, dtype=torch.float32)
    stage = 3

    if N_CTX <= 512:
        grid = (triton.cdiv(N_CTX, 128), 1, Z * H)
    else:
        grid = (Z, H, triton.cdiv(N_CTX, 128))

    _attn_fwd[grid](
        sm_scale,
        M,
        Z,
        H,
        Q,
        K,
        V,
        O,
        N_CTX=N_CTX,
        HEAD_DIM=D,
        BLOCK_M=128,
        BLOCK_N=64,
        STAGE=stage,
        num_warps=16,
        num_stages=3,
        grf_mode="auto",
    )
    return O


class Model(torch.nn.Module):
    def __init__(self, D_HEAD: int):
        super().__init__()
        self.sm_scale = 0.125
        self.causal = False
        self.D_HEAD = D_HEAD

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return attn_fwd_launch(Q, K, V, causal=self.causal, sm_scale=self.sm_scale)