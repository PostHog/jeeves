from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp


# Triton folds a float32 -> bf16 -> float32 round trip away and fuses what is left into FMAs, so rounding goes through the bits.
@triton.jit
def _round_to_bf16(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(tl.float32, bitcast=True)
    return tl.where(x != x, x, rounded)


# Rounds cos, sin, both products and their sum to bf16, like apply_rotary's eager path on bf16.
@triton.jit
def _rotary(x, cos, sin, out, heads, D: tl.constexpr, N_FREQS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
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


# The state update of fla's fused_recurrent_gated_delta_rule_fwd_kernel, with the same operations in the same order so that it rounds the
# same, run for the first steps[n] tokens of row n. It updates the state in place and computes no outputs.
@triton.jit
def _gated_delta_rule_advance(k, v, g, beta, state, steps, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                              BK: tl.constexpr, BV: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_k = k + (i_n * T * H + i_h) * K + o_k
    p_v = v + (i_n * T * HV + i_hv) * V + o_v
    p_g = g + i_n * T * HV + i_hv
    p_beta = beta + i_n * T * HV + i_hv
    p_h = state + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    for _ in tl.range(0, tl.load(steps + i_n)):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        p_k += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV
    tl.store(p_h, b_h, mask=mask_h)


def gated_delta_rule_advance(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor,
                             steps: torch.Tensor) -> None:
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if state.dtype != torch.float32 or not state.is_contiguous() or state.shape != (B, HV, K, V):
        raise ValueError(f"state: expected a contiguous float32 tensor of shape {(B, HV, K, V)}, got {state.dtype} {tuple(state.shape)}")
    # fla's block sizes, warps and stages, so that the reductions run in the same order.
    BV = min(8, triton.next_power_of_2(V))
    _gated_delta_rule_advance[(triton.cdiv(V, BV), B * HV)](k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), state, steps, T,
                                                           H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BV, num_warps=1,
                                                           num_stages=3)
