from __future__ import annotations

import torch
import triton
import triton.language as tl


# Triton folds a float32 -> bf16 -> float32 round trip away and fuses what is left into FMAs, so rounding goes through the bits.
@triton.jit
def _round_to_bf16(x):
    bits = x.to(tl.uint32, bitcast=True)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(tl.float32, bitcast=True)


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
    if 2 * n_freqs > D:
        raise ValueError(f"x: head dim {D} is smaller than the {2 * n_freqs} rotary dims")
    x = x.contiguous()
    cos, sin = (t.float().expand(B, T, n_freqs).contiguous() for t in (cos, sin))
    out = torch.empty_like(x)
    if x.numel():
        _rotary[(B * T * H,)](x, cos, sin, out, H, D=D, N_FREQS=n_freqs, BLOCK=triton.next_power_of_2(D))
    return out
