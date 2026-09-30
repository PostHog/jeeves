from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Triton folds a float32 -> bf16 -> float32 round trip away and fuses what is left into FMAs, so rounding goes through the bits.
@triton.jit
def _round_to_bf16(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(tl.float32, bitcast=True)
    return tl.where(x != x, x, rounded)


# Rounds cos, sin, both products and their sum to bf16, like apply_rotary's eager path on bf16.
@triton.jit
def _rotary(x, cos, sin, out, heads, D: tl.constexpr, N_FREQS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    token = row // heads
    d = tl.arange(0, BLOCK)
    inside = d < D
    rotated = d < 2 * N_FREQS
    lower = d < N_FREQS
    freq = tl.where(lower, d, d - N_FREQS)
    value = tl.load(x + row * D + d, mask=inside, other=0.0).to(tl.float32)
    partner = tl.load(x + row * D + tl.where(lower, d + N_FREQS, d - N_FREQS), mask=rotated, other=0.0).to(tl.float32)
    c = _round_to_bf16(tl.load(cos + token * N_FREQS + freq, mask=rotated, other=0.0))
    s = _round_to_bf16(tl.load(sin + token * N_FREQS + freq, mask=rotated, other=0.0))
    turned = tl.where(lower, -partner, partner)
    y = _round_to_bf16(value * c) + _round_to_bf16(turned * s)
    tl.store(out + row * D + d, tl.where(rotated, y, value).to(tl.bfloat16), mask=inside)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, T, H, D = x.shape
    n_freqs = cos.shape[-1]
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x: expected bfloat16, got {x.dtype}")
    if 2 * n_freqs > D:
        raise ValueError(f"x: head dim {D} is smaller than the {2 * n_freqs} rotary dims")
    x = x.contiguous()
    cos, sin = (t.float().expand(B, T, n_freqs).contiguous() for t in (cos, sin))
    out = torch.empty_like(x)
    if x.numel():
        _rotary[(B * T * H,)](x, cos, sin, out, H, D=D, N_FREQS=n_freqs, BLOCK=triton.next_power_of_2(D))
    return out


# PyTorch's CUDA sigmoid and softplus use expf, log1pf and IEEE division and keep subnormals. libdevice, div_rn and a launch with enable_reflect_ftz=False do the same; tl.exp and "/" are approximate.
@triton.jit
def _delta_gates(b, a, dt_bias, log_decay_rate, valid, beta, g, stride_b, stride_a, HV: tl.constexpr, HAS_VALID: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    inside = h < HV
    b_value = tl.load(b + row * stride_b + h, mask=inside, other=0.0).to(tl.float32)
    a_value = tl.load(a + row * stride_a + h, mask=inside, other=0.0).to(tl.float32)
    softplus_input = a_value + tl.load(dt_bias + h, mask=inside, other=0.0).to(tl.float32)
    sigmoid = _round_to_bf16(tl.math.div_rn(1.0, 1.0 + libdevice.exp(-b_value)))
    softplus = tl.where(softplus_input > 20.0, softplus_input, libdevice.log1p(libdevice.exp(softplus_input)))
    log_decay = tl.load(log_decay_rate + h, mask=inside, other=0.0) * softplus
    if HAS_VALID:
        kept = tl.load(valid + row).to(tl.float32)
        sigmoid = sigmoid * kept
        log_decay = log_decay * kept
    tl.store(beta + row * HV + h, sigmoid.to(tl.bfloat16), mask=inside)
    tl.store(g + row * HV + h, log_decay, mask=inside)


def delta_gates(b: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, log_decay_rate: torch.Tensor,
                valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, HV = b.shape
    if b.dtype != torch.bfloat16 or a.dtype != torch.bfloat16:
        raise ValueError(f"b, a: expected bfloat16, got {b.dtype}, {a.dtype}")
    if valid is not None and (valid.shape != tuple(lead) or valid.dtype != torch.bool):
        raise ValueError(f"valid: expected a bool tensor of shape {tuple(lead)}, got {valid.dtype} {tuple(valid.shape)}")
    b2, a2 = b.reshape(-1, HV), a.reshape(-1, HV)
    if b2.stride(-1) != 1 or a2.stride(-1) != 1:
        b2, a2 = b2.contiguous(), a2.contiguous()
    rows = b2.shape[0]
    beta = torch.empty(*lead, HV, dtype=torch.bfloat16, device=b.device)
    g = torch.empty(*lead, HV, dtype=torch.float32, device=b.device)
    valid_rows = None if valid is None else valid.reshape(rows).contiguous()
    if rows:
        _delta_gates[(rows,)](b2, a2, dt_bias, log_decay_rate.float().contiguous(), valid_rows, beta, g, b2.stride(0), a2.stride(0), HV=HV,
                              HAS_VALID=valid is not None, BLOCK=triton.next_power_of_2(HV), enable_reflect_ftz=False)
    return beta, g


# Flash decoding: each program attends the query rows of one key head over one slice of the keys, and a second kernel combines the slices by their log-sum-exp. A few query rows over a long cache then use the whole GPU, where SDPA's kernel walks all keys in a few thread blocks.
@triton.jit
def _decode_attention_split(q, k, v, mask, partial_o, partial_lse, scale_log2e, L, keys_per_split, splits, stride_qb, stride_qr, stride_qh,
                            stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh, stride_mb, stride_mr, HKV: tl.constexpr,
                            R: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    b, kvh = bh // HKV, bh % HKV
    rq = tl.arange(0, BLOCK_Q)
    row, head = rq // G, kvh * G + rq % G
    q_ok = rq < R * G
    d = tl.arange(0, D)
    Q = tl.load(q + b * stride_qb + row[:, None] * stride_qr + head[:, None] * stride_qh + d[None, :], mask=q_ok[:, None], other=0.0)
    m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, D], tl.float32)
    start = split * keys_per_split
    end = tl.minimum(start + keys_per_split, L)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        n_ok = n < end
        keys = tl.load(k + b * stride_kb + n[:, None] * stride_ks + kvh * stride_kh + d[None, :], mask=n_ok[:, None], other=0.0)
        bias = tl.load(mask + b * stride_mb + row[:, None] * stride_mr + n[None, :], mask=q_ok[:, None] & n_ok[None, :], other=float("-inf"))
        s = tl.dot(Q, tl.trans(keys)) * scale_log2e + bias.to(tl.float32) * 1.4426950408889634
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # A row can have every key of a slice masked; shifting by 0 keeps its terms at exp2(-inf) = 0 instead of NaN.
        m_shift = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_shift[:, None])
        alpha = tl.exp2(m_i - m_shift)
        l_i = l_i * alpha + tl.sum(p, 1)
        values = tl.load(v + b * stride_vb + n[:, None] * stride_vs + kvh * stride_vh + d[None, :], mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(values.dtype), values)
        m_i = m_new
    seen = l_i > 0.0
    slot = (bh * splits + split) * BLOCK_Q + rq
    tl.store(partial_o + slot[:, None] * D + d[None, :], acc / tl.where(seen, l_i, 1.0)[:, None])
    tl.store(partial_lse + slot, tl.where(seen, m_i + tl.log2(tl.where(seen, l_i, 1.0)), float("-inf")))


@triton.jit
def _decode_attention_combine(partial_o, partial_lse, o, splits, stride_ob, stride_or, stride_oh, HKV: tl.constexpr, G: tl.constexpr,
                              D: tl.constexpr, BLOCK_Q: tl.constexpr, SPLITS_P2: tl.constexpr):
    bh, rq = tl.program_id(0), tl.program_id(1)
    b, kvh = bh // HKV, bh % HKV
    row, head = rq // G, kvh * G + rq % G
    s = tl.arange(0, SPLITS_P2)
    s_ok = s < splits
    lse = tl.load(partial_lse + (bh * splits + s) * BLOCK_Q + rq, mask=s_ok, other=float("-inf"))
    weight = tl.exp2(lse - tl.max(lse, 0))
    d = tl.arange(0, D)
    parts = tl.load(partial_o + ((bh * splits + s[:, None]) * BLOCK_Q + rq) * D + d[None, :], mask=s_ok[:, None], other=0.0)
    out = tl.sum(parts * weight[:, None], 0) / tl.sum(weight, 0)
    tl.store(o + b * stride_ob + row * stride_or + head * stride_oh + d, out.to(o.dtype.element_ty))


def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    B, R, H, D = q.shape
    L, Hkv = mask.shape[-1], k_cache.shape[2]
    G = H // Hkv
    if q.dtype != torch.bfloat16 or H % Hkv or D & (D - 1) or R * G > 64:
        raise ValueError(f"q: expected bfloat16 with a power-of-two head dim and at most 64 query rows per key head, got {q.dtype} {tuple(q.shape)}")
    if min(t.stride(-1) for t in (q, k_cache, v_cache, mask)) != 1 or mask.shape[1] != 1:
        raise ValueError("q, k_cache, v_cache and mask need a contiguous last dim, and mask one head")
    block_n = 64
    block_q = max(16, triton.next_power_of_2(R * G))
    blocks = triton.cdiv(L, block_n)
    # Each slice writes a float32 partial of block_q x D values: many slices hide latency when the partials are small, but large partials cost more to write and combine than the latency they hide.
    programs = torch.cuda.get_device_properties(q.device).multi_processor_count * (4 if block_q * D <= 4096 else 1)
    splits = max(1, min(blocks, triton.cdiv(programs, B * Hkv)))
    keys_per_split = triton.cdiv(blocks, splits) * block_n
    splits = triton.cdiv(L, keys_per_split)
    partial_o = torch.empty(B * Hkv * splits * block_q, D, device=q.device, dtype=torch.float32)
    partial_lse = torch.empty(B * Hkv * splits * block_q, device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    _decode_attention_split[(B * Hkv, splits)](q, k_cache, v_cache, mask, partial_o, partial_lse, scale * 1.4426950408889634, L, keys_per_split,
                                               splits, *q.stride()[:3], *k_cache.stride()[:3], *v_cache.stride()[:3], mask.stride(0),
                                               mask.stride(2), HKV=Hkv, R=R, G=G, D=D, BLOCK_Q=block_q, BLOCK_N=block_n,
                                               num_warps=4 if D <= 128 else 8, num_stages=2)
    _decode_attention_combine[(B * Hkv, R * G)](partial_o, partial_lse, o, splits, *o.stride()[:3], HKV=Hkv, G=G, D=D, BLOCK_Q=block_q,
                                                SPLITS_P2=triton.next_power_of_2(splits), num_warps=4)
    return o
