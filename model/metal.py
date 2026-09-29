from __future__ import annotations

import torch

DK = DV = 128
SIMD = 32
BV = 8
WIDE_ROW_THREADS = 256
WIDE_ROW_DIM = 2048

SOURCE = f"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DK = {DK}, DV = {DV}, BV = {BV}, SIMD = {SIMD}, ROWS_PER_LANE = DK / SIMD, WIDE_ROW_THREADS = {WIDE_ROW_THREADS};
constant constexpr float Q_SCALE = {DK ** -0.5!r}f;
""" + r"""
// Metal's exp is biased low near 0 and the state takes one decay factor per token, so small |g| uses a Taylor series exact in fp32.
inline float decay_factor(float g) {
    return g > -0.0078125f ? fma(g, fma(g, fma(g, fma(g, 1.0f / 24.0f, 1.0f / 6.0f), 0.5f), 1.0f), 1.0f) : precise::exp(g);
}

template <bool OUTPUTS, bool ADVANCE>
inline void gated_delta_rule(device const bfloat* q, device const bfloat* k, device const bfloat* v, device const float* g,
                             device const bfloat* beta, device float* state, device bfloat* out, uint n_steps, uint T, uint H,
                             uint3 tg, uint lane) {
    const uint v0 = tg.x * BV, h = tg.y, b = tg.z;
    device float* state_block = state + (b * H + h) * DK * DV;
    float tile[ROWS_PER_LANE][BV];
    for (uint r = 0; r < ROWS_PER_LANE; ++r)
        for (uint c = 0; c < BV; ++c)
            tile[r][c] = state_block[(lane + SIMD * r) * DV + v0 + c];
    for (uint t = 0; t < n_steps; ++t) {
        const uint token_head = (b * T + t) * H + h;
        float kr[ROWS_PER_LANE], qr[ROWS_PER_LANE], k_sq = 0.0f, q_sq = 0.0f;
        for (uint r = 0; r < ROWS_PER_LANE; ++r) {
            kr[r] = float(k[token_head * DK + lane + SIMD * r]);
            k_sq += kr[r] * kr[r];
            if (OUTPUTS) {
                qr[r] = float(q[token_head * DK + lane + SIMD * r]);
                q_sq += qr[r] * qr[r];
            }
        }
        const float k_inv_norm = precise::rsqrt(simd_sum(k_sq) + 1e-6f);
        const float q_inv_norm = OUTPUTS ? precise::rsqrt(simd_sum(q_sq) + 1e-6f) : 0.0f;
        for (uint r = 0; r < ROWS_PER_LANE; ++r) {
            kr[r] = kr[r] * k_inv_norm;
            if (OUTPUTS)
                qr[r] = qr[r] * q_inv_norm * Q_SCALE;
        }
        const float step_decay = decay_factor(g[token_head]);
        const float step_beta = float(beta[token_head]);
        float o[BV];
        for (uint c = 0; c < BV; ++c) {
            float retrieved = 0.0f;
            for (uint r = 0; r < ROWS_PER_LANE; ++r) {
                tile[r][c] = tile[r][c] * step_decay;
                retrieved += kr[r] * tile[r][c];
            }
            const float delta = step_beta * (float(v[token_head * DV + v0 + c]) - simd_sum(retrieved));
            float oc = 0.0f;
            for (uint r = 0; r < ROWS_PER_LANE; ++r) {
                tile[r][c] = tile[r][c] + kr[r] * delta;
                if (OUTPUTS)
                    oc += qr[r] * tile[r][c];
            }
            if (OUTPUTS)
                o[c] = simd_sum(oc);
        }
        if (OUTPUTS && lane == 0)
            for (uint c = 0; c < BV; ++c)
                out[token_head * DV + v0 + c] = bfloat(o[c]);
    }
    if (ADVANCE)
        for (uint r = 0; r < ROWS_PER_LANE; ++r)
            for (uint c = 0; c < BV; ++c)
                state_block[(lane + SIMD * r) * DV + v0 + c] = tile[r][c];
}

#define OUTPUTS_KERNEL(NAME, ADVANCE) \
kernel void NAME(device const bfloat* q [[buffer(0)]], device const bfloat* k [[buffer(1)]], device const bfloat* v [[buffer(2)]], \
                 device const float* g [[buffer(3)]], device const bfloat* beta [[buffer(4)]], device float* state [[buffer(5)]], \
                 device bfloat* out [[buffer(6)]], constant long& T [[buffer(7)]], constant long& H [[buffer(8)]], \
                 uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    gated_delta_rule<true, ADVANCE>(q, k, v, g, beta, state, out, uint(T), uint(T), uint(H), tg, lane); \
}
OUTPUTS_KERNEL(gated_delta_rule_outputs, false)
OUTPUTS_KERNEL(gated_delta_rule_outputs_advance, true)

kernel void gated_delta_rule_advance(device const bfloat* k [[buffer(0)]], device const bfloat* v [[buffer(1)]],
                                     device const float* g [[buffer(2)]], device const bfloat* beta [[buffer(3)]],
                                     device float* state [[buffer(4)]], device const long* steps [[buffer(5)]],
                                     constant long& T [[buffer(6)]], constant long& H [[buffer(7)]],
                                     uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const long requested = steps[tg.z];
    const uint n_steps = requested <= 0 ? 0u : uint(requested < T ? requested : T);
    gated_delta_rule<false, true>(nullptr, k, v, g, beta, state, nullptr, n_steps, uint(T), uint(H), tg, lane);
}

// RMSNorm with weight offset 1, as in model.RMSNorm: fp32 mean of squares, rsqrt(mean + eps), times (1 + w), one bf16 rounding.
inline float rms_norm_value(float x, float inv_rms, bfloat w) {
    return (x * inv_rms) * (1.0f + float(w));
}

kernel void rms_norm(device const bfloat* x [[buffer(0)]], device const bfloat* w [[buffer(1)]], device bfloat* out [[buffer(2)]],
                     constant long& D [[buffer(3)]], constant float& eps [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat* xr = x + row * uint(D);
    float sq = 0.0f;
    for (uint i = lane; i < uint(D); i += SIMD) { const float v = float(xr[i]); sq += v * v; }
    const float inv_rms = precise::rsqrt(simd_sum(sq) / float(D) + eps);
    for (uint i = lane; i < uint(D); i += SIMD)
        out[row * uint(D) + i] = bfloat(rms_norm_value(float(xr[i]), inv_rms, w[i]));
}

kernel void rms_norm_wide(device const bfloat* x [[buffer(0)]], device const bfloat* w [[buffer(1)]], device bfloat* out [[buffer(2)]],
                          constant long& D [[buffer(3)]], constant float& eps [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[WIDE_ROW_THREADS / SIMD];
    device const bfloat* xr = x + row * uint(D);
    float sq = 0.0f;
    for (uint i = tid; i < uint(D); i += WIDE_ROW_THREADS) { const float v = float(xr[i]); sq += v * v; }
    sq = simd_sum(sq);
    if (lane == 0) partial[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < WIDE_ROW_THREADS / SIMD; ++g) total += partial[g];
    const float inv_rms = precise::rsqrt(total / float(D) + eps);
    for (uint i = tid; i < uint(D); i += WIDE_ROW_THREADS)
        out[row * uint(D) + i] = bfloat(rms_norm_value(float(xr[i]), inv_rms, w[i]));
}

// Rotary on the first 2 * half dims of each head. It rounds cos, sin, both products and the sum to bf16, like apply_rotary on bf16 tensors.
kernel void rotary(device const bfloat* x [[buffer(0)]], device const float* cos_ [[buffer(1)]], device const float* sin_ [[buffer(2)]],
                   device bfloat* out [[buffer(3)]], constant long& heads [[buffer(4)]], constant long& D [[buffer(5)]],
                   constant long& rotary_half [[buffer(6)]], uint2 pos [[thread_position_in_grid]]) {
    const uint d = pos.x, token_head = pos.y, h = uint(rotary_half);
    const uint idx = token_head * uint(D) + d;
    if (d >= 2 * h) {
        out[idx] = x[idx];
        return;
    }
    const uint freq = (token_head / uint(heads)) * h + d % h;
    const float partner = d < h ? -float(x[idx + h]) : float(x[idx - h]);
    const float c = float(bfloat(cos_[freq])), s = float(bfloat(sin_[freq]));
    out[idx] = bfloat(float(bfloat(float(x[idx]) * c)) + float(bfloat(partner * s)));
}
"""

_library = None


def library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(SOURCE)
    return _library


def _require(name: str, t: torch.Tensor, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    if t.dtype != dtype or tuple(t.shape) != shape:
        raise ValueError(f"{name}: expected {dtype} {shape}, got {t.dtype} {tuple(t.shape)}")


def _check_inputs(k, v, g, beta, state) -> tuple[int, int, int]:
    B, T, H = k.shape[:3]
    _require("k", k, torch.bfloat16, (B, T, H, DK))
    _require("v", v, torch.bfloat16, (B, T, H, DV))
    _require("g", g, torch.float32, (B, T, H))
    _require("beta", beta, torch.bfloat16, (B, T, H))
    if state.dtype != torch.float32 or not state.is_contiguous() or state.shape[0] < B or tuple(state.shape[1:]) != (H, DK, DV):
        raise ValueError(f"state: expected contiguous float32 (>={B}, {H}, {DK}, {DV}), got {state.dtype} {tuple(state.shape)}")
    return B, T, H


def _grid(B: int, H: int) -> dict:
    return {"threads": (SIMD * DV // BV, H, B), "group_size": (SIMD, 1, 1)}


def gated_delta_rule_outputs(q, k, v, g, beta, state, advance_state: bool = False) -> torch.Tensor:
    B, T, H = _check_inputs(k, v, g, beta, state)
    _require("q", q, torch.bfloat16, (B, T, H, DK))
    out = torch.empty(B, T, H, DV, dtype=torch.bfloat16, device=v.device)
    kernel = library().gated_delta_rule_outputs_advance if advance_state else library().gated_delta_rule_outputs
    kernel(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), state, out, T, H, **_grid(B, H))
    return out


def gated_delta_rule_advance_inplace(k, v, g, beta, state, steps) -> None:
    B, T, H = _check_inputs(k, v, g, beta, state)
    if steps.dtype != torch.long or steps.numel() < B:
        raise ValueError(f"steps: expected int64 with at least {B} entries, got {steps.dtype} {tuple(steps.shape)}")
    library().gated_delta_rule_advance(k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), state, steps.contiguous(),
                                       T, H, **_grid(B, H))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    D = x.shape[-1]
    _require("weight", weight, torch.bfloat16, (D,))
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x: expected torch.bfloat16, got {x.dtype}")
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    rows = x.numel() // D
    if rows == 0:
        return out
    if D >= WIDE_ROW_DIM:
        library().rms_norm_wide(x, weight, out, D, eps, threads=(WIDE_ROW_THREADS * rows, 1, 1), group_size=(WIDE_ROW_THREADS, 1, 1))
    else:
        library().rms_norm(x, weight, out, D, eps, threads=(SIMD * rows, 1, 1), group_size=(SIMD, 1, 1))
    return out


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, T, H, D = x.shape
    half = cos.shape[-1]
    if x.dtype != torch.bfloat16 or 2 * half > D:
        raise ValueError(f"x: expected bfloat16 with at least {2 * half} dims per head, got {x.dtype} {tuple(x.shape)}")
    cos, sin = (t.float().expand(B, T, half).contiguous() for t in (cos, sin))
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    if x.numel():
        library().rotary(x.contiguous(), cos, sin, out, H, D, half, threads=(D, B * T * H, 1), group_size=(min(D, 256), 1, 1))
    return out
