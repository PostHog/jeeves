from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from inference.engine import EFFICIENT, MASK_ROW_ALIGNMENT, additive_mask, merge_linears_in_place
from inference.fp8 import FP8Linear

CUDA = torch.device("cuda")


def quantized(linear: nn.Linear) -> FP8Linear:
    fp8 = FP8Linear(linear.in_features, linear.out_features, device=linear.weight.device)
    fp8.load_weight(linear.weight)
    return fp8


@torch.no_grad()
def merges() -> list[str]:
    failures = []
    torch.manual_seed(0)
    x = torch.randn(3, 16, 4096, device=CUDA, dtype=torch.bfloat16)
    parts = [nn.Linear(4096, n, bias=False, device=CUDA, dtype=torch.bfloat16) for n in (8192, 32, 32)]
    separate = [p(x) for p in parts]
    merged = merge_linears_in_place(tuple(parts))
    for i, (a, b) in enumerate(zip(merged(x).split([p.out_features for p in parts], dim=-1), separate)):
        if not torch.allclose(a.float(), b.float(), rtol=2 ** -7, atol=1e-2):
            failures.append(f"merged bf16 projection {i} differs from its separate GEMM by more than bf16 rounding")
    if any(p.weight.untyped_storage().data_ptr() != merged.weight.untyped_storage().data_ptr() for p in parts):
        failures.append("the parts of a merged bf16 projection are not views of the merged weight")
    if merge_linears_in_place((nn.Linear(4096, 1024, device=CUDA), nn.Linear(4096, 1024, device=CUDA))) is not None:
        failures.append("merge_linears_in_place merged projections with a bias")
    fp8_parts = [quantized(nn.Linear(4096, n, bias=False, device=CUDA, dtype=torch.bfloat16)) for n in (4096, 1024, 1024)]
    codes, scales = [p.weight.clone() for p in fp8_parts], [p.scale.clone() for p in fp8_parts]
    fp8_merged = merge_linears_in_place(tuple(fp8_parts))
    for p, c, s in zip(fp8_parts, codes, scales):
        if not torch.equal(p.weight.view(torch.uint8), c.view(torch.uint8)) or not torch.equal(p.scale, s):
            failures.append("merging fp8 projections changed a part's codes or scales")
    for i, (a, p) in enumerate(zip(fp8_merged(x).split([p.out_features for p in fp8_parts], dim=-1), fp8_parts)):
        if not torch.allclose(a.float(), p(x).float(), rtol=2 ** -7, atol=1e-2):
            failures.append(f"merged fp8 projection {i} differs from its separate GEMM by more than bf16 rounding")
    if merge_linears_in_place((fp8_parts[0], parts[1])) is not None:
        failures.append("merge_linears_in_place merged fp8 and bf16 projections")
    return failures


@torch.no_grad()
def masks() -> list[str]:
    failures = []
    torch.manual_seed(0)
    for B, Q, H, Hkv, D, L in ((1, 16, 16, 4, 256, 524), (3, 16, 16, 4, 256, 1040), (2, 12, 32, 32, 128, 524)):
        q = torch.randn(B, Q, H, D, device=CUDA, dtype=torch.bfloat16)
        k, v = (torch.randn(B, L, Hkv, D, device=CUDA, dtype=torch.bfloat16) for _ in range(2))
        allowed = torch.rand(B, 1, Q, L, device=CUDA) < 0.5
        allowed[..., 0] = True
        mask = additive_mask(allowed, torch.bfloat16)
        if mask.stride(-2) % MASK_ROW_ALIGNMENT:
            failures.append(f"additive_mask for {L} keys has a row stride of {mask.stride(-2)}")
        outputs = []
        for m in (allowed, mask):
            with sdpa_kernel(EFFICIENT):
                outputs.append(F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=m,
                                                              enable_gqa=H != Hkv))
        if not torch.equal(*outputs):
            failures.append(f"SDPA with additive_mask {(B, Q, H, Hkv, D, L)} is not bitwise equal to SDPA with the boolean mask")
    return failures


@torch.no_grad()
def fp8_matmuls() -> list[str]:
    failures = []
    torch.manual_seed(1)
    # 1040 columns are not a multiple of BLOCK_N, and 17408 columns fill the GPU without split-K.
    for N, K in ((4096, 4096), (1024, 12288), (1040, 4096), (17408, 4096)):
        linear = quantized(nn.Linear(K, N, bias=False, device=CUDA, dtype=torch.bfloat16))
        dequantized = linear.weight.float() * linear.scale[:, None]
        for M in (1, 4, 12, 16, 17, 48, 256, 300, 1024):
            x = torch.randn(M, K, device=CUDA, dtype=torch.bfloat16)
            # An outlier dominates its row's outputs, so only rows without one show errors in the rest of the GEMM, and tiny rows fall into fp16's subnormals unless scaled up.
            x[::2, 0] = 1e5
            x[1::4] *= 2 ** -20
            reference = x.float() @ dequantized.t()
            out = linear(x)
            error = (out.float() - reference).norm(dim=1) / reference.norm(dim=1)
            if not out.isfinite().all() or not error.max() <= 2e-3:
                failures.append(f"FP8Linear {N}x{K} at {M} rows differs from float32 by up to {error.max():.2e} in a row, more than bf16 rounding")
            if not torch.equal(out, linear(x)):
                failures.append(f"FP8Linear {N}x{K} at {M} rows gives different results on a second call")
    return failures


def main() -> None:
    failures = merges() + masks() + fp8_matmuls()
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
