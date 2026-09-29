from __future__ import annotations

import torch

DK = DV = 128
SIMD = 32
BV = 8

SOURCE = f"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DK = {DK}, DV = {DV}, BV = {BV}, SIMD = {SIMD}, ROWS_PER_LANE = DK / SIMD;
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
