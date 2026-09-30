from __future__ import annotations

import sys

import torch

from model.model import apply_rotary, delta_gates, gated_delta_rule_advance, set_kernels

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


@torch.no_grad()
def delta_advance() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(5)
    for B, T, H, HV, D in ((4, 4, 32, 32, 128), (2, 4, 16, 32, 128), (3, 1, 32, 32, 128)):
        k = torch.randn(B, T, H, D, generator=gen).to(torch.bfloat16).to(CUDA)
        v = torch.randn(B, T, HV, D, generator=gen).to(torch.bfloat16).to(CUDA)
        g = -torch.rand(B, T, HV, generator=gen).to(CUDA)
        beta = torch.rand(B, T, HV, generator=gen).to(torch.bfloat16).to(CUDA)
        state = torch.randn(B, HV, D, D, generator=gen).to(CUDA)
        steps = torch.arange(B).remainder(T + 1).to(CUDA)

        def advanced() -> torch.Tensor:
            advanced_state = state.clone()
            gated_delta_rule_advance(k, v, g, beta, advanced_state, steps)
            return advanced_state
        eager, out = eager_then_triton(advanced)
        if not torch.equal(out, eager):
            failures.append(f"gated_delta_rule_advance {(B, T, H, HV, D)} with the Triton kernel is not bitwise equal to fla's masked steps")
        if not torch.equal(out[steps == 0], state[steps == 0]):
            failures.append(f"gated_delta_rule_advance {(B, T, H, HV, D)} changed a row that advances 0 steps")
    return failures


@torch.no_grad()
def gates() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(6)
    HV = 32
    merged = (torch.randn(3, 5, 64 + 2 * HV, generator=gen) * 12).to(torch.bfloat16).to(CUDA)
    b, a = merged[..., 64:64 + HV], merged[..., 64 + HV:]
    rate = -torch.rand(HV, generator=gen).exp().to(CUDA)
    keep = (torch.rand(3, 5, 1, generator=gen) < 0.7).float().to(CUDA)
    for dt_bias in (torch.randn(HV, generator=gen).to(CUDA), torch.randn(HV, generator=gen).to(torch.bfloat16).to(CUDA)):
        for kept in (None, keep):
            eager, out = eager_then_triton(lambda: delta_gates(b, a, dt_bias, rate, kept))
            for name, x, y in zip(("beta", "g"), out, eager):
                if x.dtype != y.dtype or not torch.equal(x, y):
                    failures.append(f"delta_gates {name} (dt_bias {dt_bias.dtype}, keep {kept is not None}) with the Triton kernel is not "
                                    f"bitwise equal to eager")
    return failures


def main() -> None:
    failures = rotary() + delta_advance() + gates()
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
