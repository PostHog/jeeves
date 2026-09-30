from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from drafter.view import DrafterView
from inference.fp8_checkpoint import load_fp8_view
from inference.fp8_weights import FP8, FP8_MAX, quantize_rows
from loader.dataloader import Encoder
from model import PointerHead

CONFIGS = [triton.Config({"BLOCK_N": n, "BLOCK_M": m, "BLOCK_K": k}, num_warps=w, num_stages=st)
           for n in (32, 64, 128) for m in (16, 32, 64, 128) for k in (128, 256) for w, st in ((4, 4), (8, 3))
           if not (n == 128 and m == 128 and k == 256)]
LARGE_M = 256
TARGET_BLOCKS = 264


@triton.autotune(configs=CONFIGS, key=["MB", "N", "K", "SPLIT"], cache_results=True)
@triton.jit
def w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_wn, stride_ym, MB,
                 SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    part = tl.program_id(2)
    chunk = K // SPLIT
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(part * chunk, (part + 1) * chunk, BLOCK_K):
        kk = k0 + rk
        w = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :], mask=rn[:, None] < N, other=0.0)
        x = tl.load(x_ptr + rm[None, :] * stride_xm + kk[:, None], mask=rm[None, :] < M, other=0.0)
        acc += tl.dot(w.to(tl.bfloat16), x)
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * s[:, None]
    out = y_ptr + part * M * N + rm[None, :] * stride_ym + rn[:, None]
    if SPLIT == 1:
        tl.store(out, acc.to(tl.bfloat16), mask=(rm[None, :] < M) & (rn[:, None] < N))
    else:
        tl.store(out, acc, mask=(rm[None, :] < M) & (rn[:, None] < N))


def bucket(m: int) -> int:
    return 1 << max(4, (m - 1).bit_length())


def split_for(N: int, K: int) -> int:
    split = 1
    while triton.cdiv(N, 64) * split < TARGET_BLOCKS and K % (split * 2 * 256) == 0 and K // (split * 2) >= 1024:
        split *= 2
    return split


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: torch.device | str | None = None):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8, device=device))
        self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False) if bias else None
        self.in_features, self.out_features = in_features, out_features
        self.split = split_for(self.out_features, self.in_features)

    @torch.no_grad()
    def load_codes(self, codes: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight.copy_(codes)
        self.scale.copy_(scale)

    @torch.no_grad()
    def load_weight(self, weight: torch.Tensor) -> None:
        self.load_codes(*quantize_rows(weight.to(self.weight.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        M, K = x2.shape
        N = self.out_features
        if M > LARGE_M:
            sx = x2.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
            y = torch._scaled_mm((x2.float() / sx).to(FP8), self.weight.t(), scale_a=sx, scale_b=self.scale[None],
                                 out_dtype=torch.bfloat16)
            if self.bias is not None:
                y = y + self.bias
            return y.view(*shape[:-1], N)
        split = self.split
        y = torch.empty(split, M, N, device=x.device, dtype=torch.bfloat16 if split == 1 else torch.float32)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(M, meta["BLOCK_M"]), split)
        w8a16_kernel[grid](x2, self.weight, self.scale, y, M, N, K, x2.stride(0), self.weight.stride(0), N, bucket(M), SPLIT=split)
        y = y[0] if split == 1 else y.sum(0).to(torch.bfloat16)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], N)


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
            for rows in (16, 32, 64, 128, 256):
                m(torch.zeros(rows, m.in_features, device=w.device, dtype=torch.bfloat16))
    return len(seen)

