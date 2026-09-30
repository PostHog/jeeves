from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from inference.fp8_weights import FP8, FP8_MAX, quantize_rows, replace_linears

LARGE_M = 256
# Above this many rows, one conversion kernel for the activations costs less than converting them again in every program.
CONVERT_IN_GEMM_M = 16
TARGET_BLOCKS = 264
BLOCK_N, BLOCK_K = 64, 128


# Hopper's tensor cores multiply fp8 only by fp8, so the weights go to fp16, which holds every e4m3 value exactly and takes one conversion
# instruction (bf16 takes several). Activations go to fp16 after a power of two that keeps them inside its range, which is also exact.
@triton.jit
def _fp16_shift(amax):
    exponent = ((amax.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32) - 127
    return tl.maximum(exponent - 13, 0)


@triton.jit
def _power_of_two(exponent):
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fp16_rows(x, xh, row_scale, K, stride_xm, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        kk = k0 + tl.arange(0, BLOCK)
        amax = tl.maximum(amax, tl.abs(tl.load(x + row * stride_xm + kk, mask=kk < K, other=0.0).to(tl.float32)))
    shift = _fp16_shift(tl.max(amax, 0))
    for k0 in range(0, K, BLOCK):
        kk = k0 + tl.arange(0, BLOCK)
        value = tl.load(x + row * stride_xm + kk, mask=kk < K, other=0.0).to(tl.float32)
        tl.store(xh + row * K + kk, (value * _power_of_two(-shift)).to(tl.float16), mask=kk < K)
    tl.store(row_scale + row, _power_of_two(shift))


# Split-K runs in one launch: the last program to finish a tile adds the float32 partials in split order, so results do not depend on
# scheduling, and resets the tile's counter for the next launch.
@triton.jit
def _fp8_matmul(x, row_scale, w, w_scale, y, partial, counters, M, N, K, stride_xm, stride_wn, stride_ym, k_per_split, SPLIT: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, CONVERT: tl.constexpr):
    pid_n, pid_m, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k_start = split * k_per_split
    for k0 in range(k_start, k_start + k_per_split, BLOCK_K):
        kk = k0 + rk
        wt = tl.load(w + rn[:, None] * stride_wn + kk[None, :], mask=rn[:, None] < N, other=0.0).to(tl.float16)
        xt = tl.load(x + rm[None, :] * stride_xm + kk[:, None], mask=rm[None, :] < M, other=0.0)
        if CONVERT:
            shift = _fp16_shift(tl.max(tl.abs(xt.to(tl.float32)), 0))
            xh = (xt.to(tl.float32) * _power_of_two(-shift)[None, :]).to(tl.float16)
            acc += tl.dot(wt, xh) * _power_of_two(shift)[None, :]
        else:
            acc += tl.dot(wt, xt)
    if not CONVERT:
        acc *= tl.load(row_scale + rm, mask=rm < M, other=0.0)[None, :]
    out_mask = (rm[None, :] < M) & (rn[:, None] < N)
    w_s = tl.load(w_scale + rn, mask=rn < N, other=0.0)
    if SPLIT == 1:
        tl.store(y + rm[None, :] * stride_ym + rn[:, None], (acc * w_s[:, None]).to(tl.bfloat16), mask=out_mask)
    else:
        tile = pid_m * tl.num_programs(0) + pid_n
        offsets = rm[None, :] * N + rn[:, None]
        tl.store(partial + split * M * N + offsets, acc, mask=out_mask)
        if tl.atomic_add(counters + tile, 1, sem="acq_rel") == SPLIT - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for part in range(SPLIT):
                total += tl.load(partial + part * M * N + offsets, mask=out_mask, other=0.0, cache_modifier=".cg")
            tl.store(y + rm[None, :] * stride_ym + rn[:, None], (total * w_s[:, None]).to(tl.bfloat16), mask=out_mask)
            tl.store(counters + tile, 0)


TILE_COUNTERS: dict[torch.device, torch.Tensor] = {}


def split_for(N: int, K: int) -> int:
    split = 1
    while triton.cdiv(N, BLOCK_N) * split < TARGET_BLOCKS and K % (split * 2 * BLOCK_K) == 0 and K // (split * 2) >= 1024:
        split *= 2
    return split


def fp8_matmul(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, split: int) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    block_m = min(64, max(16, triton.next_power_of_2(M)))
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, block_m), split)
    counters = TILE_COUNTERS.get(x.device)
    if counters is None:
        counters = TILE_COUNTERS[x.device] = torch.zeros(1 << 16, dtype=torch.int32, device=x.device)
    if grid[0] * grid[1] > counters.numel():
        raise ValueError(f"{grid[0] * grid[1]} tiles exceed the {counters.numel()} tile counters")
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    partial = torch.empty(split, M, N, device=x.device, dtype=torch.float32) if split > 1 else y
    convert = M <= CONVERT_IN_GEMM_M
    if convert:
        source, row_scale = x, y
    else:
        source = torch.empty(M, K, device=x.device, dtype=torch.float16)
        row_scale = torch.empty(M, device=x.device, dtype=torch.float32)
        _fp16_rows[(M,)](x, source, row_scale, K, x.stride(0), BLOCK=1024)
    _fp8_matmul[grid](source, row_scale, weight, scale, y, partial, counters, M, N, K, source.stride(0), weight.stride(0), y.stride(0),
                      K // split, SPLIT=split, BLOCK_N=BLOCK_N, BLOCK_M=block_m, BLOCK_K=BLOCK_K, CONVERT=convert, num_warps=4,
                      num_stages=4)
    return y


class FP8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        self.out_features, self.in_features = weight.shape
        self.split = split_for(self.out_features, self.in_features)

    @classmethod
    def quantized(cls, linear: nn.Linear) -> FP8Linear:
        return cls(*quantize_rows(linear.weight), linear.bias)

    @classmethod
    def concatenated(cls, parts: list[FP8Linear]) -> FP8Linear:
        merged = cls(torch.cat([p.weight for p in parts]), torch.cat([p.scale for p in parts]), None)
        sizes = [p.out_features for p in parts]
        for p, weight, scale in zip(parts, merged.weight.split(sizes), merged.scale.split(sizes)):
            p.weight, p.scale = weight, scale
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        M = x2.shape[0]
        N = self.out_features
        if M > LARGE_M:
            sx = x2.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
            y = torch._scaled_mm((x2.float() / sx).to(FP8), self.weight.t(), scale_a=sx, scale_b=self.scale[None],
                                 out_dtype=torch.bfloat16)
        else:
            y = fp8_matmul(x2, self.weight, self.scale, self.split)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], N)


def quantize(module: nn.Module) -> int:
    return replace_linears(module, FP8Linear.quantized)


@torch.no_grad()
def warm(module: nn.Module) -> int:
    seen = set()
    for m in module.modules():
        if isinstance(m, FP8Linear) and (m.out_features, m.in_features) not in seen:
            seen.add((m.out_features, m.in_features))
            w = m.weight
            for rows in (4, 12, 16, 32, 64, 128, 256):
                m(torch.zeros(rows, m.in_features, device=w.device, dtype=torch.bfloat16))
    return len(seen)
