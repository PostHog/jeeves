from __future__ import annotations

import sys

import torch
import torch.nn as nn

from inference.engine import merge_linears_in_place
from inference.fp8 import FP8Linear

CUDA = torch.device("cuda")


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
    fp8_parts = [FP8Linear.quantized(nn.Linear(4096, n, bias=False, device=CUDA, dtype=torch.bfloat16)) for n in (4096, 1024, 1024)]
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


def main() -> None:
    failures = merges()
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
