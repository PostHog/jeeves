from __future__ import annotations

import sys

import torch

from model.model import apply_rotary, set_kernels

CUDA = torch.device("cuda")


def eager_then_triton(fn):
    set_kernels(triton_inference=False)
    try:
        eager = fn()
    finally:
        set_kernels(triton_inference=True)
    return eager, fn()


@torch.no_grad()
def rotary() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(4)
    for B, T, H, D in ((1, 4, 32, 128), (1, 12, 32, 128), (4, 16, 16, 256), (2, 190, 4, 256), (1, 1, 32, 128)):
        x = torch.randn(B, T, H, D, generator=gen).to(torch.bfloat16).to(CUDA)
        positions = (torch.arange(T)[None] + torch.randint(0, 8000, (B, 1), generator=gen)).float()[..., None]
        freqs = positions * torch.logspace(0, -7, 32)
        cos, sin = freqs.cos().to(CUDA), freqs.sin().to(CUDA)
        eager, out = eager_then_triton(lambda: apply_rotary(x, cos, sin))
        if not torch.equal(out, eager):
            failures.append(f"apply_rotary {(B, T, H, D)} with the Triton kernel is not bitwise equal to eager")
        shared_eager, shared_out = eager_then_triton(lambda: apply_rotary(x, cos[0], sin[0]))
        if not torch.equal(shared_out, shared_eager):
            failures.append(f"apply_rotary {(B, T, H, D)} with 2-D cos changed with the Triton kernel on")
    x = torch.randn(1, 2, 4, 128, generator=gen).to(torch.bfloat16).to(CUDA)
    x[0, 0, 0, 5] = float("nan")
    freqs = torch.arange(2).float()[None, :, None] * torch.logspace(0, -7, 32)
    eager, out = eager_then_triton(lambda: apply_rotary(x, freqs.cos().to(CUDA), freqs.sin().to(CUDA)))
    if not torch.equal(out.isnan(), eager.isnan()):
        failures.append("apply_rotary with the Triton kernel does not propagate NaN like eager")
    return failures


def main() -> None:
    failures = rotary()
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
