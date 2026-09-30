from __future__ import annotations

import torch

SIMD = 32

_HEADER = f"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD = {SIMD};
""" + r"""
// Attention visits the used cache positions [0, used) and then the tail [tail_start, L); the positions in between are always masked.
inline uint visited_position(uint i, uint used, uint tail_start) {
    return i < used ? i : tail_start + (i - used);
}
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


# Gated RMSNorm: one simdgroup per row, as in model.GatedRMSNorm.

_GATED_RMS_NORM = r"""
// Must round like model.GatedRMSNorm's eager path: the normalised value, its product with the weight, and the product with silu(z).
// MPS evaluates silu(z) as z / (1 + exp(-z)); the other algebraic forms round differently.
kernel void gated_rms_norm(device const bfloat* x [[buffer(0)]], device const bfloat* z [[buffer(1)]], device const bfloat* w [[buffer(2)]],
                           device bfloat* out [[buffer(3)]], constant long& D [[buffer(4)]], constant float& eps [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat* xr = x + row * uint(D);
    device const bfloat* zr = z + row * uint(D);
    float sq = 0.0f;
    for (uint i = lane; i < uint(D); i += SIMD) { const float v = float(xr[i]); sq += v * v; }
    const float inv_rms = precise::rsqrt(simd_sum(sq) / float(D) + eps);
    for (uint i = lane; i < uint(D); i += SIMD) {
        const bfloat normed = bfloat(float(xr[i]) * inv_rms);
        const bfloat weighted = bfloat(float(w[i]) * float(normed));
        const float zf = float(zr[i]);
        out[row * uint(D) + i] = bfloat(float(weighted) * (zf / (1.0f + precise::exp(-zf))));
    }
}
"""


def gated_rms_norm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    D = x.shape[-1]
    _require("x", x, torch.bfloat16)
    _require("z", z, torch.bfloat16, tuple(x.shape))
    _require("weight", weight, torch.bfloat16, (D,))
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    rows = x.numel() // D if D else 0
    if rows:
        library().gated_rms_norm(x.contiguous(), z.contiguous(), weight.contiguous(), out, D, eps, threads=(SIMD * rows, 1, 1), group_size=(SIMD, 1, 1))
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
                             device const long* used_end, device bfloat* out, uint Q, uint H, uint HKV, uint L, uint tail_start,
                             uint row_stride, float scale, uint3 tg, uint sg, uint lane, threadgroup float* part_max,
                             threadgroup float* part_sum, threadgroup float* part_out) {
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
    const uint used = min(uint(used_end[b]), tail_start), visited = used + (L - tail_start);
    const uint share = (visited + ATTENTION_SPLITS - 1) / ATTENTION_SPLITS, first = sg * share, last = min(visited, first + share);
    float m = -INFINITY, s = 0.0f;
    for (uint i = first; i < last; ++i) {
        const uint l = visited_position(i, used, tail_start);
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
                                 device const bfloat* v [[buffer(2)]], device const bool* mask [[buffer(3)]], device const long* used_end [[buffer(4)]], \
                                 device bfloat* out [[buffer(5)]], constant long& Q [[buffer(6)]], constant long& H [[buffer(7)]], \
                                 constant long& HKV [[buffer(8)]], constant long& L [[buffer(9)]], constant long& tail_start [[buffer(10)]], \
                                 constant long& row_stride [[buffer(11)]], constant float& scale [[buffer(12)]], \
                                 uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                                 uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float part_max[ATTENTION_SPLITS], part_sum[ATTENTION_SPLITS], part_out[ATTENTION_SPLITS * D]; \
    cached_attention<D>(q, k, v, mask, used_end, out, uint(Q), uint(H), uint(HKV), uint(L), uint(tail_start), uint(row_stride), scale, tg, sg, \
                        lane, part_max, part_sum, part_out); \
}
""" + "".join(f"CACHED_ATTENTION_KERNEL({D})\n" for D in ATTENTION_HEAD_DIMS)


def _visited_extent(B: int, length: int, used_end: torch.Tensor | None, tail_start: int | None, device) -> tuple[torch.Tensor, int]:
    if used_end is None:
        return torch.full((B,), length, dtype=torch.long, device=device), length
    _require("used_end", used_end, torch.long, (B,))
    if not 0 <= tail_start <= length:
        raise ValueError(f"tail_start {tail_start} is outside [0, {length}]")
    return used_end.contiguous(), tail_start


def cached_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, mask: torch.Tensor, scale: float,
                     used_end: torch.Tensor | None = None, tail_start: int | None = None) -> torch.Tensor:
    B, Q, H, D = q.shape
    length = mask.shape[-1]
    HKV = k_cache.shape[2]
    _require("q", q, torch.bfloat16)
    _require("mask", mask, torch.bool, (B, Q, length))
    used_end, tail_start = _visited_extent(B, length, used_end, tail_start, q.device)
    for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        _require_buffer(name, cache, torch.bfloat16, at_least=(B, length), then=(HKV, D))
    if D not in ATTENTION_HEAD_DIMS or H % HKV:
        raise ValueError(f"unsupported head layout: {H} query heads, {HKV} key/value heads, head dim {D}")
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    getattr(library(), f"cached_attention_{D}")(q.contiguous(), k_cache, v_cache, mask.contiguous(), used_end, out, Q, H, HKV, length, tail_start,
                                                k_cache.stride(0), scale, threads=(SIMD * ATTENTION_SPLITS * Q, H, B),
                                                group_size=(SIMD * ATTENTION_SPLITS, 1, 1))
    return out



# Multi-query attention for several rows: each threadgroup takes one (row, kv head, chunk of positions) and the query rows that share that
# kv head, so a chunk of the cache is read once for all of them. merge_attention_chunks then combines the chunks' partial softmaxes.

CHUNK_POSITIONS = 256
MULTI_QUERY_CONFIGS = {128: (2, 6, True), 256: (4, 7, False)}

_MULTI_QUERY_ATTENTION = f"""
constant constexpr uint CHUNK_POSITIONS = {CHUNK_POSITIONS};
""" + r"""
inline float4 bf16x4(uint2 w) {
    return float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xffff0000u), as_type<float>(w.y << 16), as_type<float>(w.y & 0xffff0000u));
}

// Scores: lane i owns position l0 + i of each 32-position block (K rows come from threadgroup memory when STAGE_K, padded against bank
// conflicts, or straight from device memory). Values: lanes split D, so value rows are read coalesced. Simdgroup s owns QPS query rows.
template <uint D, uint QPS, uint NSG, bool STAGE_K>
inline void multi_query_chunk(device const bfloat* q, device const bfloat* k, device const bfloat* v, device const bool* mask,
                              device const long* used_end, device float* part_max, device float* part_sum, device float* part_out, uint Q,
                              uint H, uint HKV, uint L, uint tail_start, uint row_stride, float scale, uint n_chunks, uint n_row_groups,
                              uint3 tg, uint sg, uint lane, uint tid,
                              threadgroup float* qs, threadgroup bfloat* ks) {
    constexpr uint DIMS_PER_LANE = D / SIMD, K_PITCH = D + 8, ROWS_PER_GROUP = NSG * QPS;
    const uint chunk = tg.x / n_row_groups, row_group = tg.x % n_row_groups, kv_head = tg.y, b = tg.z;
    const uint heads_per_kv = H / HKV, G = Q * heads_per_kv, r0 = row_group * ROWS_PER_GROUP + sg * QPS;
    threadgroup float* my_q = qs + sg * QPS * D;
    for (uint j = 0; j < QPS; ++j) {
        const uint r = r0 + j, qi = r / heads_per_kv, h = kv_head * heads_per_kv + r % heads_per_kv;
        for (uint d = lane; d < D; d += SIMD)
            my_q[j * D + d] = r < G ? float(q[((b * Q + qi) * H + h) * D + d]) : 0.0f;
    }
    device const bfloat* kb = k + b * row_stride + kv_head * D;
    device const bfloat* vb = v + b * row_stride + kv_head * D + lane * DIMS_PER_LANE;
    const uint used = min(uint(used_end[b]), tail_start), visited = used + (L - tail_start);
    const uint position_stride = HKV * D, c0 = chunk * CHUNK_POSITIONS, c1 = min(visited, c0 + CHUNK_POSITIONS);
    float m[QPS], s[QPS], o[QPS][DIMS_PER_LANE];
    for (uint j = 0; j < QPS; ++j) {
        m[j] = -INFINITY;
        s[j] = 0.0f;
        for (uint d = 0; d < DIMS_PER_LANE; ++d)
            o[j][d] = 0.0f;
    }
    if (!STAGE_K)
        simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint l0 = c0; l0 < c1; l0 += SIMD) {
        const uint n = min(SIMD, c1 - l0);
        if (STAGE_K) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < n * (D / 8); e += NSG * SIMD) {
                const uint row = e / (D / 8), col = (e % (D / 8)) * 8;
                const uint position = visited_position(l0 + row, used, tail_start);
                *(threadgroup uint4*)(ks + row * K_PITCH + col) = *(device const uint4*)(kb + position * position_stride + col);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint l = visited_position(l0 + lane, used, tail_start);
        const bool in_chunk = lane < n;
        float score[QPS];
        for (uint j = 0; j < QPS; ++j)
            score[j] = 0.0f;
        if (in_chunk) {
            for (uint c = 0; c < D / 4; ++c) {
                const float4 kf = STAGE_K ? bf16x4(((threadgroup const uint2*)(ks + lane * K_PITCH))[c])
                                          : bf16x4(((device const uint2*)(kb + l * position_stride))[c]);
                for (uint j = 0; j < QPS; ++j)
                    score[j] += dot(*(threadgroup const float4*)(my_q + j * D + c * 4), kf);
            }
        }
        float p[QPS];
        for (uint j = 0; j < QPS; ++j) {
            const uint r = r0 + j, qi = r < G ? r / heads_per_kv : 0;
            const bool allowed = in_chunk && r < G && mask[(b * Q + qi) * L + l];
            const float sc = allowed ? score[j] * scale : -INFINITY;
            const float m_new = max(m[j], simd_max(sc));
            if (m_new == -INFINITY) {
                p[j] = 0.0f;
                continue;
            }
            const float rescale = precise::exp(m[j] - m_new);
            p[j] = allowed ? precise::exp(sc - m_new) : 0.0f;
            s[j] = s[j] * rescale + simd_sum(p[j]);
            for (uint d = 0; d < DIMS_PER_LANE; ++d)
                o[j][d] *= rescale;
            m[j] = m_new;
        }
        for (uint i = 0; i < n; ++i) {
            float vf[DIMS_PER_LANE];
            for (uint d = 0; d < DIMS_PER_LANE; ++d)
                vf[d] = float(vb[visited_position(l0 + i, used, tail_start) * position_stride + d]);
            for (uint j = 0; j < QPS; ++j) {
                const float pj = simd_shuffle(p[j], i);
                for (uint d = 0; d < DIMS_PER_LANE; ++d)
                    o[j][d] += pj * vf[d];
            }
        }
    }
    for (uint j = 0; j < QPS; ++j) {
        const uint r = r0 + j;
        if (r >= G)
            continue;
        const uint idx = ((b * HKV + kv_head) * G + r) * n_chunks + chunk;
        if (lane == 0) {
            part_max[idx] = m[j];
            part_sum[idx] = s[j];
        }
        for (uint d = 0; d < DIMS_PER_LANE; ++d)
            part_out[idx * D + lane * DIMS_PER_LANE + d] = o[j][d];
    }
}

// One simdgroup per (row, kv head, query row); a query with no allowed position gets 0, as SDPA returns on MPS.
template <uint D>
inline void merge_attention_chunks(device const float* part_max, device const float* part_sum, device const float* part_out, device bfloat* out,
                                   uint Q, uint H, uint HKV, uint n_chunks, uint3 tg, uint lane) {
    constexpr uint DIMS_PER_LANE = D / SIMD;
    const uint r = tg.x, kv_head = tg.y, b = tg.z;
    const uint heads_per_kv = H / HKV, G = Q * heads_per_kv, qi = r / heads_per_kv, h = kv_head * heads_per_kv + r % heads_per_kv;
    const uint base = ((b * HKV + kv_head) * G + r) * n_chunks;
    float m_all = -INFINITY;
    for (uint c = 0; c < n_chunks; ++c)
        m_all = max(m_all, part_max[base + c]);
    float s_all = 0.0f, o_all[DIMS_PER_LANE];
    for (uint d = 0; d < DIMS_PER_LANE; ++d)
        o_all[d] = 0.0f;
    for (uint c = 0; c < n_chunks; ++c) {
        if (part_max[base + c] == -INFINITY)
            continue;
        const float w = precise::exp(part_max[base + c] - m_all);
        s_all += part_sum[base + c] * w;
        for (uint d = 0; d < DIMS_PER_LANE; ++d)
            o_all[d] += part_out[(base + c) * D + lane * DIMS_PER_LANE + d] * w;
    }
    for (uint d = 0; d < DIMS_PER_LANE; ++d)
        out[((b * Q + qi) * H + h) * D + lane * DIMS_PER_LANE + d] = bfloat(s_all > 0.0f ? o_all[d] / s_all : 0.0f);
}

#define MULTI_QUERY_KERNELS(D, QPS, NSG, STAGE_K) \
kernel void multi_query_chunk_##D(device const bfloat* q [[buffer(0)]], device const bfloat* k [[buffer(1)]], device const bfloat* v [[buffer(2)]], \
        device const bool* mask [[buffer(3)]], device float* part_max [[buffer(4)]], device float* part_sum [[buffer(5)]], \
        device float* part_out [[buffer(6)]], device const long* used_end [[buffer(7)]], constant long& Q [[buffer(8)]], \
        constant long& H [[buffer(9)]], constant long& HKV [[buffer(10)]], constant long& L [[buffer(11)]], constant long& tail_start [[buffer(12)]], \
        constant long& row_stride [[buffer(13)]], constant float& scale [[buffer(14)]], constant long& n_chunks [[buffer(15)]], \
        constant long& n_row_groups [[buffer(16)]], uint3 tg [[threadgroup_position_in_grid]], \
        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float qs[QPS * NSG * D]; \
    threadgroup bfloat ks[STAGE_K ? SIMD * (D + 8) : 1]; \
    multi_query_chunk<D, QPS, NSG, STAGE_K>(q, k, v, mask, used_end, part_max, part_sum, part_out, uint(Q), uint(H), uint(HKV), uint(L), \
                                            uint(tail_start), uint(row_stride), scale, uint(n_chunks), uint(n_row_groups), tg, sg, lane, tid, qs, ks); \
} \
kernel void merge_attention_chunks_##D(device const float* part_max [[buffer(0)]], device const float* part_sum [[buffer(1)]], \
        device const float* part_out [[buffer(2)]], device bfloat* out [[buffer(3)]], constant long& Q [[buffer(4)]], constant long& H [[buffer(5)]], \
        constant long& HKV [[buffer(6)]], constant long& n_chunks [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    merge_attention_chunks<D>(part_max, part_sum, part_out, out, uint(Q), uint(H), uint(HKV), uint(n_chunks), tg, lane); \
}
""" + "".join(f"MULTI_QUERY_KERNELS({D}, {qps}, {nsg}, {'true' if stage else 'false'})\n" for D, (qps, nsg, stage) in MULTI_QUERY_CONFIGS.items())


def multi_query_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, mask: torch.Tensor, scale: float,
                          used_end: torch.Tensor | None = None, tail_start: int | None = None) -> torch.Tensor:
    B, Q, H, D = q.shape
    length = mask.shape[-1]
    HKV = k_cache.shape[2]
    _require("q", q, torch.bfloat16)
    _require("mask", mask, torch.bool, (B, Q, length))
    used_end, tail_start = _visited_extent(B, length, used_end, tail_start, q.device)
    for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        _require_buffer(name, cache, torch.bfloat16, at_least=(B, length), then=(HKV, D))
    if D not in MULTI_QUERY_CONFIGS or H % HKV:
        raise ValueError(f"unsupported head layout: {H} query heads, {HKV} key/value heads, head dim {D}")
    qps, nsg, _ = MULTI_QUERY_CONFIGS[D]
    G = Q * (H // HKV)
    n_row_groups = -(-G // (qps * nsg))
    n_chunks = -(-length // CHUNK_POSITIONS)
    part_max = torch.empty(B * HKV * G * n_chunks, dtype=torch.float32, device=q.device)
    part_sum = torch.empty_like(part_max)
    part_out = torch.empty(B * HKV * G * n_chunks * D, dtype=torch.float32, device=q.device)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    getattr(library(), f"multi_query_chunk_{D}")(q.contiguous(), k_cache, v_cache, mask.contiguous(), part_max, part_sum, part_out, used_end, Q,
                                                  H, HKV, length, tail_start, k_cache.stride(0), scale, n_chunks, n_row_groups,
                                                  threads=(n_chunks * n_row_groups * nsg * SIMD, HKV, B), group_size=(nsg * SIMD, 1, 1))
    getattr(library(), f"merge_attention_chunks_{D}")(part_max, part_sum, part_out, out, Q, H, HKV, n_chunks,
                                                       threads=(G * SIMD, HKV, B), group_size=(SIMD, 1, 1))
    return out


# FP8 linear (w8a16): y = x @ (codes * scale).T with float accumulation and one rounding to bf16, the math of inference/fp8.py's
# w8a16 kernel. A simdgroup owns FP8_TILE_FEATURES output features and FP8_TILE_ROWS rows of x.

FP8_TILE_FEATURES = 16
FP8_TILE_ROWS = 8
FP8_BLOCK_K = 64
FP8_SIMDGROUPS = 4
FP8_K_SPLITS = 4

_FP8_LINEAR = f"""
constant constexpr uint FP8_TILE_FEATURES = {FP8_TILE_FEATURES}, FP8_BLOCK_K = {FP8_BLOCK_K};
""" + r"""
// Four e4m3fn codes as their values times 2^-8: the sign moves to fp16 bit 15 and the exponent and mantissa to bits 13..7, which
// is exact for every code, subnormals included. even gets bytes 0 and 2 of word, odd gets bytes 1 and 3.
inline void fp8x4_times_2_pow_minus_8(uint word, thread float2& even, thread float2& odd) {
    even = float2(as_type<half2>(((word & 0x00800080u) << 8) | ((word & 0x007F007Fu) << 7)));
    odd = float2(as_type<half2>((word & 0x80008000u) | ((word & 0x7F007F00u) >> 1)));
}

// Each lane loads 16 contiguous code bytes of one feature per 64-k block and uses bytes 2q and 2q + 1 as its fragment of k tile q.
// The x fragments follow the same permutation of k within the block, so the dot products are unchanged.
template <bool PARTIAL>
inline void fp8_linear_tile(device const bfloat* x, device const uchar* w, device const float* scale, device void* y,
                            uint M, uint N, uint K, uint k_per_part, uint3 tg, uint sg, uint sgs, uint lane) {
    const uint n0 = (tg.x * sgs + sg) * FP8_TILE_FEATURES;
    if (n0 >= N) return;
    const uint m0 = tg.z * 8, part = tg.y;
    const uint quad = lane / 4;
    const uint frag_row = (quad & 4) + ((lane / 2) % 4);
    const uint frag_col = (quad & 2) * 2 + (lane % 2) * 2;
    const uint x_offset = 8 * (frag_row & ~1u) + (frag_row & 1u);
    device const bfloat* x0 = x + ulong(min(m0 + frag_col, M - 1)) * K + x_offset;
    device const bfloat* x1 = x + ulong(min(m0 + frag_col + 1, M - 1)) * K + x_offset;
    device const uchar* w0 = w + ulong(n0 + frag_row) * K + 8 * frag_col;
    device const uchar* w1 = w0 + 8 * ulong(K);
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f), acc1 = simdgroup_float8x8(0.0f);
    for (uint k0 = part * k_per_part; k0 < (part + 1) * k_per_part; k0 += FP8_BLOCK_K) {
        const uint4 codes0 = *(device const uint4*)(w0 + k0);
        const uint4 codes1 = *(device const uint4*)(w1 + k0);
        for (uint p = 0; p < 4; ++p) {
            float2 even0, odd0, even1, odd1;
            fp8x4_times_2_pow_minus_8(codes0[p], even0, odd0);
            fp8x4_times_2_pow_minus_8(codes1[p], even1, odd1);
            for (uint h = 0; h < 2; ++h) {
                const uint k = k0 + 4 * p + 2 * h;
                simdgroup_float8x8 b, a0, a1;
                b.thread_elements()[0] = float(x0[k]);
                b.thread_elements()[1] = float(x1[k]);
                a0.thread_elements()[0] = h ? even0.y : even0.x;
                a0.thread_elements()[1] = h ? odd0.y : odd0.x;
                a1.thread_elements()[0] = h ? even1.y : even1.x;
                a1.thread_elements()[1] = h ? odd1.y : odd1.x;
                simdgroup_multiply_accumulate(acc0, a0, b, acc0);
                simdgroup_multiply_accumulate(acc1, a1, b, acc1);
            }
        }
    }
    for (uint t = 0; t < 2; ++t) {
        const uint n = n0 + 8 * t + frag_row;
        for (uint e = 0; e < 2; ++e) {
            const uint m = m0 + frag_col + e;
            if (m >= M) continue;
            const float v = t ? acc1.thread_elements()[e] : acc0.thread_elements()[e];
            if (PARTIAL)
                ((device float*)y)[(ulong(part) * M + m) * N + n] = v;
            else
                ((device bfloat*)y)[ulong(m) * N + n] = bfloat(v * (scale[n] * 256.0f));
        }
    }
}

kernel void fp8_linear(device const bfloat* x [[buffer(0)]], device const uchar* w [[buffer(1)]], device const float* scale [[buffer(2)]],
                       device bfloat* y [[buffer(3)]], constant long& M [[buffer(4)]], constant long& N [[buffer(5)]],
                       constant long& K [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    fp8_linear_tile<false>(x, w, scale, y, uint(M), uint(N), uint(K), uint(K), tg, sg, sgs, lane);
}

kernel void fp8_linear_partial(device const bfloat* x [[buffer(0)]], device const uchar* w [[buffer(1)]], device const float* scale [[buffer(2)]],
                               device float* parts [[buffer(3)]], constant long& M [[buffer(4)]], constant long& N [[buffer(5)]],
                               constant long& K [[buffer(6)]], constant long& k_per_part [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    fp8_linear_tile<true>(x, w, scale, parts, uint(M), uint(N), uint(K), uint(k_per_part), tg, sg, sgs, lane);
}

kernel void fp8_linear_merge(device const float* parts [[buffer(0)]], device const float* scale [[buffer(1)]], device bfloat* y [[buffer(2)]],
                             constant long& MN [[buffer(3)]], constant long& N [[buffer(4)]], constant long& splits [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= uint(MN)) return;
    float acc = 0.0f;
    for (uint p = 0; p < uint(splits); ++p)
        acc += parts[ulong(p) * ulong(MN) + i];
    y[i] = bfloat(acc * (scale[i % uint(N)] * 256.0f));
}
"""


def fp8_linear(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    N, K = codes.shape
    _require("x", x, torch.bfloat16)
    _require("codes", codes, torch.uint8, (N, K))
    _require("scale", scale, torch.float32, (N,))
    if x.shape[-1] != K or K % (FP8_BLOCK_K * FP8_K_SPLITS) or N % FP8_TILE_FEATURES or not codes.is_contiguous() or codes.storage_offset() % 16:
        raise ValueError(f"fp8_linear: needs x (..., {K}), K a multiple of {FP8_BLOCK_K * FP8_K_SPLITS}, N a multiple of {FP8_TILE_FEATURES} "
                         f"and 16-byte aligned contiguous codes; got x {tuple(x.shape)}, codes {tuple(codes.shape)}")
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    out = torch.empty(*x.shape[:-1], N, dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return out
    row_tiles = -(-M // FP8_TILE_ROWS)
    groups = -(-N // (FP8_TILE_FEATURES * FP8_SIMDGROUPS))
    simdgroups = row_tiles * N // FP8_TILE_FEATURES
    # Measured on an M4 Pro: splitting k pays off once there are too few simdgroups to keep the GPU busy through the whole k loop.
    splits = FP8_K_SPLITS if simdgroups <= K // 8 else 1
    grid = {"threads": (groups * FP8_SIMDGROUPS * SIMD, splits, row_tiles), "group_size": (FP8_SIMDGROUPS * SIMD, 1, 1)}
    scale = scale.contiguous()
    if splits == 1:
        library().fp8_linear(x2, codes, scale, out, M, N, K, **grid)
        return out
    parts = torch.empty(splits, M, N, dtype=torch.float32, device=x.device)
    library().fp8_linear_partial(x2, codes, scale, parts, M, N, K, K // splits, **grid)
    library().fp8_linear_merge(parts, scale, out, M * N, N, splits, threads=(M * N, 1, 1), group_size=(256, 1, 1))
    return out


SOURCE = _HEADER + _GATED_DELTA_RULE + _RMS_NORM + _GATED_RMS_NORM + _ROTARY + _CACHED_ATTENTION + _MULTI_QUERY_ATTENTION + _FP8_LINEAR
_library = None


def library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(SOURCE)
    return _library
