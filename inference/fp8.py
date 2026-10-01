from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
import triton
import triton.language as tl

from drafter.view import DrafterView
from inference.fp8_checkpoint import load_fp8_view
from inference.fp8_weights import FP8, quantize_rows
from loader.dataloader import Encoder
from model import PointerHead
from model.triton_kernels import sm_count

# Above this many rows, one conversion kernel for the activations costs less than converting them again in every program.
CONVERT_IN_GEMM_MAX_ROWS = 16
# Above this many rows the row tiles alone fill the GPU, so passes skip split-K and use taller tiles that reuse each weight tile more.
SPLIT_K_MAX_ROWS = 256
# Each extra split costs a float32 partial and its reduction, so K is split only until there are this many programs per SM: in a sweep on an H100 PCIe (2026-10-01), more programs were slower and fewer left bandwidth unused. The split sets the summation order, so fp8 outputs differ between GPUs with different SM counts.
MIN_PROGRAMS_PER_SM = 1.5
MIN_K_PER_SPLIT = 1024
# A deeper pipeline has a longer prologue, which only a long K loop pays back.
DEEP_PIPELINE_STEPS = 32
BLOCK_N, BLOCK_K = 64, 128
TILE_COUNTER_SLOTS = 1 << 16
ROW_CONVERT_BLOCK = 1024
# The power of two scales the largest value of each row (of each K tile when the GEMM converts the activations itself) into [2^13, 2^14) where float32 allows, so no input, even float32, can overflow fp16's 65504, and bf16 values down to 2^-30 times the largest stay exact.
FP16_TOP_EXPONENT = tl.constexpr(13)
# One row count for each variant of _fp8_matmul that Triton compiles: tile heights 16, 32, 64 and 128, each with a row count that is and is not a multiple of 16, and 1, which Triton compiles as a constant.
WARM_ROWS = (1, 8, 16, 24, 32, 56, 64, SPLIT_K_MAX_ROWS + 8, SPLIT_K_MAX_ROWS + 16)


@triton.jit
def _fp16_shift(amax):
    exponent = ((amax.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32) - 127
    return tl.maximum(exponent - FP16_TOP_EXPONENT, -126)


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


# Split-K runs in one launch: the last program to finish a tile adds the float32 partials in split order, so results do not depend on scheduling, and resets the tile's counter for the next launch.
@triton.jit
def _fp8_matmul(x, row_scale, w, w_scale, y, partial, counters, M, N, stride_xm, stride_wn, stride_ym, k_per_split, SPLIT: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, CONVERT: tl.constexpr):
    pid_n, pid_m, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k_start = split * k_per_split
    for k0 in range(k_start, k_start + k_per_split, BLOCK_K):
        kk = k0 + rk
        # Hopper's tensor cores multiply fp8 only by fp8, so the weights go to fp16, which holds every e4m3 value exactly and takes one conversion instruction (bf16 takes several).
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
        tl.store(partial + split.to(tl.int64) * M * N + offsets, acc, mask=out_mask)
        # Triton 3.8 runs a scalar atomic in one thread and puts no barrier before a release, so without this another warp's partial could still be in flight.
        tl.debug_barrier()
        if tl.atomic_add(counters + tile, 1, sem="acq_rel") == SPLIT - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for part in range(SPLIT):
                total += tl.load(partial + part.to(tl.int64) * M * N + offsets, mask=out_mask, other=0.0, cache_modifier=".cg")
            tl.store(y + rm[None, :] * stride_ym + rn[:, None], (total * w_s[:, None]).to(tl.bfloat16), mask=out_mask)
            tl.store(counters + tile, 0)


TILE_COUNTERS: dict[tuple[torch.device, int], torch.Tensor] = {}


@lru_cache
def split_for(N: int, K: int, sms: int) -> int:
    steps = K // BLOCK_K
    splits = [s for s in range(1, steps + 1) if steps % s == 0 and K // s >= min(K, MIN_K_PER_SPLIT)]
    return next((s for s in splits if triton.cdiv(N, BLOCK_N) * s >= MIN_PROGRAMS_PER_SM * sms), splits[-1])


def fp8_matmul(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    if K == 0 or K % BLOCK_K or weight.shape[1] != K:
        raise ValueError(f"the fp8 GEMM needs in_features divisible by {BLOCK_K} and equal to the weight's, got {K} for a weight of shape "
                         f"{tuple(weight.shape)}")
    if M > SPLIT_K_MAX_ROWS:
        block_m, split, stages = 128, 1, 3
    else:
        block_m, split = min(64, max(16, triton.next_power_of_2(M))), split_for(N, K, sm_count(x.device))
        stages = 4 if K // split // BLOCK_K >= DEEP_PIPELINE_STEPS else 3
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, block_m), split)
    counters = partial = None
    if split > 1:
        # Launches on different streams can overlap and must not share tile counters; a graph keeps the counters of the stream it was captured on, so graphs captured on one stream must not replay at the same time.
        key = (x.device, torch.cuda.current_stream(x.device).cuda_stream)
        counters = TILE_COUNTERS.get(key)
        if counters is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("run an fp8 GEMM on this stream before capturing a CUDA graph on it, so that the stream's tile counters exist "
                                   "outside the graph")
            counters = TILE_COUNTERS[key] = torch.zeros(TILE_COUNTER_SLOTS, dtype=torch.int32, device=x.device)
        if grid[0] * grid[1] > counters.numel():
            raise ValueError(f"{grid[0] * grid[1]} tiles exceed the {counters.numel()} tile counters")
        partial = torch.empty(split, M, N, device=x.device, dtype=torch.float32)
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    convert = M <= CONVERT_IN_GEMM_MAX_ROWS
    if convert:
        source, row_scale = x, None
    else:
        source = torch.empty(M, K, device=x.device, dtype=torch.float16)
        row_scale = torch.empty(M, device=x.device, dtype=torch.float32)
        _fp16_rows[(M,)](x, source, row_scale, K, x.stride(0), BLOCK=ROW_CONVERT_BLOCK)
    _fp8_matmul[grid](source, row_scale, weight, scale, y, partial, counters, M, N, source.stride(0), weight.stride(0), y.stride(0),
                      K // split, SPLIT=split, BLOCK_N=BLOCK_N, BLOCK_M=block_m, BLOCK_K=BLOCK_K, CONVERT=convert, num_warps=4,
                      num_stages=stages)
    return y


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: torch.device | str | None = None):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8, device=device))
        self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False) if bias else None
        self.in_features, self.out_features = in_features, out_features

    @torch.no_grad()
    def load_codes(self, codes: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight.copy_(codes)
        self.scale.copy_(scale)

    @torch.no_grad()
    def load_weight(self, weight: torch.Tensor) -> None:
        self.load_codes(*quantize_rows(weight.to(self.weight.device)))

    @classmethod
    def concatenated(cls, parts: list[FP8Linear]) -> FP8Linear:
        sizes = [p.out_features for p in parts]
        merged = cls(parts[0].in_features, sum(sizes), device="meta")
        merged.weight, merged.scale = torch.cat([p.weight for p in parts]), torch.cat([p.scale for p in parts])
        for p, weight, scale in zip(parts, merged.weight.split(sizes), merged.scale.split(sizes)):
            p.weight, p.scale = weight, scale
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        y = fp8_matmul(x2, self.weight, self.scale)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.out_features)


def load_view(model: str, drafter: str, block: int, device: torch.device) -> tuple[DrafterView, PointerHead, Encoder]:
    view, head, encoder = load_fp8_view(model, drafter, block, device, FP8Linear)
    warm(view)
    return view, head, encoder


@torch.no_grad()
def warm(module: nn.Module) -> int:
    seen = set()
    for m in module.modules():
        if isinstance(m, FP8Linear) and (m.out_features, m.in_features) not in seen:
            seen.add((m.out_features, m.in_features))
            w = m.weight
            for rows in WARM_ROWS:
                m(torch.zeros(rows, m.in_features, device=w.device, dtype=torch.bfloat16))
    return len(seen)
