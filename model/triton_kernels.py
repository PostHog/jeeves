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
