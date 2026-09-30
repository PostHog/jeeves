from __future__ import annotations

import sys

import torch

from model import triton_kernels
from model.model import apply_rotary, set_kernels

CUDA = torch.device("cuda")


@torch.no_grad()
def rotary() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(4)
    for B, T, H, D in ((1, 4, 32, 128), (1, 12, 32, 128), (4, 16, 16, 256), (2, 190, 4, 256), (1, 1, 32, 128)):
        x = torch.randn(B, T, H, D, generator=gen).to(torch.bfloat16).to(CUDA)
        positions = (torch.arange(T)[None] + torch.randint(0, 8000, (B, 1), generator=gen)).float()[..., None]
        freqs = positions * torch.logspace(0, -7, 32)
        cos, sin = freqs.cos().to(CUDA), freqs.sin().to(CUDA)
        set_kernels(triton_inference=False)
        eager = apply_rotary(x, cos, sin)
        set_kernels(triton_inference=True)
        out = triton_kernels.rotary(x, cos, sin)
        if not torch.equal(out, eager):
            failures.append(f"rotary {(B, T, H, D)} in Triton is not bitwise equal to eager")
    return failures


def main() -> None:
    failures = rotary()
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
