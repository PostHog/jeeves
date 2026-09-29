from __future__ import annotations

import torch

DK, DV, BV = 128, 128, 8

SOURCE = """
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DK = 128, DV = 128, BV = 8, R = DK / 32;

kernel void gated_delta_rule(
    device const bfloat* q [[buffer(0)]], device const bfloat* k [[buffer(1)]], device const bfloat* v [[buffer(2)]],
    device const float* g [[buffer(3)]], device const bfloat* beta [[buffer(4)]], device float* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]], device const long* steps [[buffer(7)]], constant uint& T [[buffer(8)]],
    constant uint& H [[buffer(9)]], constant uint& write_state [[buffer(10)]], constant uint& write_out [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    const uint v0 = tg.x * BV, h = tg.y, b = tg.z;
    device float* S = state + (b * H + h) * DK * DV;
    float s[R][BV];
    for (uint r = 0; r < R; ++r)
        for (uint c = 0; c < BV; ++c)
            s[r][c] = S[(lane + 32 * r) * DV + v0 + c];
    const uint n = uint(steps[b]);
    for (uint t = 0; t < n; ++t) {
        const uint row = (b * T + t) * H + h;
        float kr[R], qr[R], ks = 0.0f, qs = 0.0f;
        for (uint r = 0; r < R; ++r) {
            kr[r] = float(k[row * DK + lane + 32 * r]);
            qr[r] = float(q[row * DK + lane + 32 * r]);
            ks += kr[r] * kr[r];
            qs += qr[r] * qr[r];
        }
        const float kn = precise::rsqrt(simd_sum(ks) + 1e-6f);
        const float qn = precise::rsqrt(simd_sum(qs) + 1e-6f);
        for (uint r = 0; r < R; ++r) {
            kr[r] = kr[r] * kn;
            qr[r] = qr[r] * qn * 0.08838834764831845f;
        }
        const float decay = precise::exp(g[row]);
        const float bt = float(beta[row]);
        float o[BV];
        for (uint c = 0; c < BV; ++c) {
            float retrieved = 0.0f;
            for (uint r = 0; r < R; ++r) {
                s[r][c] = s[r][c] * decay;
                retrieved += kr[r] * s[r][c];
            }
            const float delta = bt * (float(v[row * DV + v0 + c]) - simd_sum(retrieved));
            float oc = 0.0f;
            for (uint r = 0; r < R; ++r) {
                s[r][c] = s[r][c] + kr[r] * delta;
                oc += qr[r] * s[r][c];
            }
            o[c] = simd_sum(oc);
        }
        if (write_out && lane == 0)
            for (uint c = 0; c < BV; ++c)
                out[row * DV + v0 + c] = bfloat(o[c]);
    }
    if (write_state)
        for (uint r = 0; r < R; ++r)
            for (uint c = 0; c < BV; ++c)
                S[(lane + 32 * r) * DV + v0 + c] = s[r][c];
}
"""

_library = None


def library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(SOURCE)
    return _library


def _run(q, k, v, g, beta, state, out, steps, write_state: bool, write_out: bool) -> None:
    B, T, H, dk = k.shape
    assert dk == DK and v.shape[-1] == DV and state.is_contiguous() and state.shape[1:] == (H, DK, DV)
    library().gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(), g.float().contiguous(), beta.contiguous(), state, out,
                               steps, T, H, int(write_state), int(write_out), threads=(32 * DV // BV, H, B), group_size=(32, 1, 1))


def gated_delta_rule_outputs(q, k, v, g, beta, state, steps, advance_state: bool = False) -> torch.Tensor:
    out = torch.empty(v.shape, dtype=torch.bfloat16, device=v.device)
    _run(q, k, v, g, beta, state, out, steps, write_state=advance_state, write_out=True)
    return out


def gated_delta_rule_advance_(k, v, g, beta, state, steps) -> None:
    _run(k, k, v, g, beta, state, v, steps, write_state=True, write_out=False)
