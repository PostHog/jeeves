from __future__ import annotations

import torch

SIMD = 32

_HEADER = f"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD = {SIMD};
"""


def _require(name: str, t: torch.Tensor, dtype: torch.dtype, shape: tuple[int, ...] | None = None) -> None:
    if t.dtype != dtype or (shape is not None and tuple(t.shape) != shape):
        raise ValueError(f"{name}: expected {dtype} {shape or 'any shape'}, got {t.dtype} {tuple(t.shape)}")


def _require_buffer(name: str, t: torch.Tensor, dtype: torch.dtype, at_least: tuple[int, ...], then: tuple[int, ...]) -> None:
    lead = tuple(t.shape[:len(at_least)])
    if t.dtype != dtype or not t.is_contiguous() or tuple(t.shape[len(at_least):]) != then or any(n < m for n, m in zip(lead, at_least)):
        raise ValueError(f"{name}: expected contiguous {dtype} with leading dims >= {at_least} and then {then}, got {t.dtype} {tuple(t.shape)}")


# Gated delta rule: one simdgroup per (row, head, block of BV value columns), holding that block of the fp32 state in registers.

DK = DV = 128
BV = 8

_GATED_DELTA_RULE = f"""
constant constexpr uint DK = {DK}, DV = {DV}, BV = {BV}, ROWS_PER_LANE = DK / SIMD;
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


def _check_delta_inputs(k, v, g, beta, state) -> tuple[int, int, int]:
    B, T, H = k.shape[:3]
    _require("k", k, torch.bfloat16, (B, T, H, DK))
    _require("v", v, torch.bfloat16, (B, T, H, DV))
    _require("g", g, torch.float32, (B, T, H))
    _require("beta", beta, torch.bfloat16, (B, T, H))
    _require_buffer("state", state, torch.float32, at_least=(B,), then=(H, DK, DV))
    return B, T, H


def _delta_grid(B: int, H: int) -> dict:
    return {"threads": (SIMD * DV // BV, H, B), "group_size": (SIMD, 1, 1)}


def gated_delta_rule_outputs(q, k, v, g, beta, state, advance_state: bool = False) -> torch.Tensor:
    B, T, H = _check_delta_inputs(k, v, g, beta, state)
    _require("q", q, torch.bfloat16, (B, T, H, DK))
    out = torch.empty(B, T, H, DV, dtype=torch.bfloat16, device=v.device)
    kernel = library().gated_delta_rule_outputs_advance if advance_state else library().gated_delta_rule_outputs
    kernel(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), state, out, T, H, **_delta_grid(B, H))
    return out


def gated_delta_rule_advance_inplace(k, v, g, beta, state, steps) -> None:
    B, T, H = _check_delta_inputs(k, v, g, beta, state)
    if steps.dtype != torch.long or steps.numel() < B:
        raise ValueError(f"steps: expected int64 with at least {B} entries, got {steps.dtype} {tuple(steps.shape)}")
    library().gated_delta_rule_advance(k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), state, steps.contiguous(),
                                       T, H, **_delta_grid(B, H))


# RMSNorm: one simdgroup per row, or one threadgroup per row once rows are wide enough to leave a simdgroup mostly looping.

WIDE_ROW_THREADS = 256
WIDE_ROW_MIN_DIM = 2048

_RMS_NORM = f"""
constant constexpr uint WIDE_ROW_THREADS = {WIDE_ROW_THREADS};
""" + r"""
// Must round like model.RMSNorm's eager path: fp32 math and one bf16 rounding, so only the order of the sum of squares differs.
inline bfloat rms_norm_value(bfloat x, float inv_rms, bfloat w) {
    return bfloat((float(x) * inv_rms) * (1.0f + float(w)));
}

kernel void rms_norm(device const bfloat* x [[buffer(0)]], device const bfloat* w [[buffer(1)]], device bfloat* out [[buffer(2)]],
                     constant long& D [[buffer(3)]], constant float& eps [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat* xr = x + row * uint(D);
    float sq = 0.0f;
    for (uint i = lane; i < uint(D); i += SIMD) { const float v = float(xr[i]); sq += v * v; }
    const float inv_rms = precise::rsqrt(simd_sum(sq) / float(D) + eps);
    for (uint i = lane; i < uint(D); i += SIMD)
        out[row * uint(D) + i] = rms_norm_value(xr[i], inv_rms, w[i]);
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
        out[row * uint(D) + i] = rms_norm_value(xr[i], inv_rms, w[i]);
}
"""


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    D = x.shape[-1]
    _require("x", x, torch.bfloat16)
    _require("weight", weight, torch.bfloat16, (D,))
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    rows = x.numel() // D if D else 0
    if rows == 0:
        return out
    if D >= WIDE_ROW_MIN_DIM:
        library().rms_norm_wide(x, weight.contiguous(), out, D, eps, threads=(WIDE_ROW_THREADS * rows, 1, 1), group_size=(WIDE_ROW_THREADS, 1, 1))
    else:
        library().rms_norm(x, weight.contiguous(), out, D, eps, threads=(SIMD * rows, 1, 1), group_size=(SIMD, 1, 1))
    return out


# Rotary: one thread per element of x.

ROTARY_GROUP = 256

_ROTARY = r"""
// Rotates the first 2 * n_freqs dims of each head and rounds cos, sin, both products and the sum to bf16, like apply_rotary on bf16.
kernel void rotary(device const bfloat* x [[buffer(0)]], device const float* cos_ [[buffer(1)]], device const float* sin_ [[buffer(2)]],
                   device bfloat* out [[buffer(3)]], constant long& heads [[buffer(4)]], constant long& D [[buffer(5)]],
                   constant long& n_freqs [[buffer(6)]], uint2 pos [[thread_position_in_grid]]) {
    const uint d = pos.x, token_head = pos.y, f = uint(n_freqs);
    const uint idx = token_head * uint(D) + d;
    if (d >= 2 * f) {
        out[idx] = x[idx];
        return;
    }
    const uint freq = (token_head / uint(heads)) * f + d % f;
    const float partner = d < f ? -float(x[idx + f]) : float(x[idx - f]);
    const float c = float(bfloat(cos_[freq])), s = float(bfloat(sin_[freq]));
    out[idx] = bfloat(float(bfloat(float(x[idx]) * c)) + float(bfloat(partner * s)));
}
"""


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, T, H, D = x.shape
    n_freqs = cos.shape[-1]
    _require("x", x, torch.bfloat16)
    if 2 * n_freqs > D:
        raise ValueError(f"x: head dim {D} is smaller than the {2 * n_freqs} rotary dims")
    cos, sin = (t.float().expand(B, T, n_freqs).contiguous() for t in (cos, sin))
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    if x.numel():
        library().rotary(x.contiguous(), cos, sin, out, H, D, n_freqs, threads=(D, B * T * H, 1), group_size=(min(D, ROTARY_GROUP), 1, 1))
    return out


# Cached attention: a few queries per row attend over a (row, position, kv head, D) cache in place.

ATTENTION_SPLITS = 2
ATTENTION_HEAD_DIMS = (128, 256)

_CACHED_ATTENTION = f"""
constant constexpr uint ATTENTION_SPLITS = {ATTENTION_SPLITS};
""" + r"""
// One threadgroup per (row, head, query). Each simdgroup runs an online softmax over its share of the positions, with D split across
// lanes, and simdgroup 0 merges the shares. A query with no allowed position gets 0, as SDPA returns on MPS.
template <uint D>
inline void cached_attention(device const bfloat* q, device const bfloat* k, device const bfloat* v, device const bool* mask,
                             device bfloat* out, uint Q, uint H, uint HKV, uint L, uint row_stride, float scale, uint3 tg, uint sg,
                             uint lane, threadgroup float* part_max, threadgroup float* part_sum, threadgroup float* part_out) {
    constexpr uint DIMS_PER_LANE = D / SIMD;
    const uint qi = tg.x, h = tg.y, b = tg.z;
    const uint kv_head = h / (H / HKV), out_base = ((b * Q + qi) * H + h) * D + lane * DIMS_PER_LANE;
    float qv[DIMS_PER_LANE], o[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; ++i) {
        qv[i] = float(q[out_base + i]);
        o[i] = 0.0f;
    }
    device const bfloat* kb = k + b * row_stride + kv_head * D + lane * DIMS_PER_LANE;
    device const bfloat* vb = v + b * row_stride + kv_head * D + lane * DIMS_PER_LANE;
    device const bool* allowed = mask + (b * Q + qi) * L;
    const uint share = (L + ATTENTION_SPLITS - 1) / ATTENTION_SPLITS, first = sg * share, last = min(L, first + share);
    float m = -INFINITY, s = 0.0f;
    for (uint l = first; l < last; ++l) {
        if (!allowed[l])
            continue;
        float dot = 0.0f;
        for (uint i = 0; i < DIMS_PER_LANE; ++i)
            dot += qv[i] * float(kb[l * HKV * D + i]);
        const float score = simd_sum(dot) * scale;
        const float m_new = max(m, score);
        const float rescale = precise::exp(m - m_new), p = precise::exp(score - m_new);
        s = s * rescale + p;
        for (uint i = 0; i < DIMS_PER_LANE; ++i)
            o[i] = o[i] * rescale + p * float(vb[l * HKV * D + i]);
        m = m_new;
    }
    if (lane == 0) {
        part_max[sg] = m;
        part_sum[sg] = s;
    }
    for (uint i = 0; i < DIMS_PER_LANE; ++i)
        part_out[sg * D + lane * DIMS_PER_LANE + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0)
        return;
    float m_all = -INFINITY;
    for (uint g = 0; g < ATTENTION_SPLITS; ++g)
        m_all = max(m_all, part_max[g]);
    float s_all = 0.0f, o_all[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; ++i)
        o_all[i] = 0.0f;
    for (uint g = 0; g < ATTENTION_SPLITS; ++g) {
        if (part_max[g] == -INFINITY)
            continue;
        const float w = precise::exp(part_max[g] - m_all);
        s_all += part_sum[g] * w;
        for (uint i = 0; i < DIMS_PER_LANE; ++i)
            o_all[i] += part_out[g * D + lane * DIMS_PER_LANE + i] * w;
    }
    for (uint i = 0; i < DIMS_PER_LANE; ++i)
        out[out_base + i] = bfloat(s_all > 0.0f ? o_all[i] / s_all : 0.0f);
}

#define CACHED_ATTENTION_KERNEL(D) \
kernel void cached_attention_##D(device const bfloat* q [[buffer(0)]], device const bfloat* k [[buffer(1)]], \
                                 device const bfloat* v [[buffer(2)]], device const bool* mask [[buffer(3)]], device bfloat* out [[buffer(4)]], \
                                 constant long& Q [[buffer(5)]], constant long& H [[buffer(6)]], constant long& HKV [[buffer(7)]], \
                                 constant long& L [[buffer(8)]], constant long& row_stride [[buffer(9)]], constant float& scale [[buffer(10)]], \
                                 uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                                 uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float part_max[ATTENTION_SPLITS], part_sum[ATTENTION_SPLITS], part_out[ATTENTION_SPLITS * D]; \
    cached_attention<D>(q, k, v, mask, out, uint(Q), uint(H), uint(HKV), uint(L), uint(row_stride), scale, tg, sg, lane, \
                        part_max, part_sum, part_out); \
}
""" + "".join(f"CACHED_ATTENTION_KERNEL({D})\n" for D in ATTENTION_HEAD_DIMS)


def cached_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    B, Q, H, D = q.shape
    length = mask.shape[-1]
    HKV = k_cache.shape[2]
    _require("q", q, torch.bfloat16)
    _require("mask", mask, torch.bool, (B, Q, length))
    for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        _require_buffer(name, cache, torch.bfloat16, at_least=(B, length), then=(HKV, D))
    if D not in ATTENTION_HEAD_DIMS or H % HKV:
        raise ValueError(f"unsupported head layout: {H} query heads, {HKV} key/value heads, head dim {D}")
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    getattr(library(), f"cached_attention_{D}")(q.contiguous(), k_cache, v_cache, mask.contiguous(), out, Q, H, HKV, length, k_cache.stride(0),
                                                scale, threads=(SIMD * ATTENTION_SPLITS * Q, H, B), group_size=(SIMD * ATTENTION_SPLITS, 1, 1))
    return out


SOURCE = _HEADER + _GATED_DELTA_RULE + _RMS_NORM + _ROTARY + _CACHED_ATTENTION
_library = None


def library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(SOURCE)
    return _library
