from __future__ import annotations

import sys

import torch

from model import metal
from model.model import RMSNorm, apply_rotary, set_kernels, torch_recurrent_gated_delta_rule

MPS = torch.device("mps")


def inputs(B: int, T: int, H: int = 32, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, T, H, 128, generator=gen).to(torch.bfloat16) for _ in range(3))
    head_rate = torch.logspace(-5, 0, H)[torch.randperm(H, generator=gen)]
    g = -head_rate * torch.rand(B, T, H, generator=gen) * 2
    beta = torch.rand(B, T, H, generator=gen).to(torch.bfloat16)
    state = torch.randn(B, H, 128, 128, generator=gen) * 0.1
    return q, k, v, g, beta, state


def exact_recurrence(q, k, v, g, beta, state):
    q, k, v, g, beta, S = (t.double() for t in (q, k, v, g, beta, state))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    outs = []
    for t in range(q.shape[1]):
        S = S * g[:, t].exp()[..., None, None]
        delta = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(outs, 1), S


def mismatch(out: torch.Tensor, exact: torch.Tensor) -> float:
    return (out.cpu() != exact.float().to(torch.bfloat16)).float().mean().item()


def rel_err(state: torch.Tensor, exact: torch.Tensor) -> float:
    return ((state.cpu().double() - exact).norm() / exact.norm()).item()


def accuracy(B: int, T: int) -> list[str]:
    q, k, v, g, beta, state = inputs(B, T)
    exact_o, exact_s = exact_recurrence(q, k, v, g, beta, state)
    cpu_o, cpu_s = torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=state, output_final_state=True)
    mps_state = state.to(MPS)
    out = metal.gated_delta_rule_outputs(*(t.to(MPS) for t in (q, k, v, g, beta)), mps_state, advance_state=True)
    line = (f"B={B} T={T}: bf16 mismatch metal {mismatch(out, exact_o):.5f} cpu-fp32 {mismatch(cpu_o, exact_o):.5f}, "
            f"state rel err metal {rel_err(mps_state, exact_s):.2e} cpu-fp32 {rel_err(cpu_s, exact_s):.2e}")
    print(line)
    failures = []
    if mismatch(out, exact_o) > 1.5 * mismatch(cpu_o, exact_o) + 1e-4:
        failures.append(f"output rounding worse than fp32 CPU: {line}")
    if rel_err(mps_state, exact_s) > 2 * rel_err(cpu_s, exact_s) + 1e-7:
        failures.append(f"state error worse than fp32 CPU: {line}")
    return failures


def exactness() -> list[str]:
    failures = []
    B, T = 3, 7
    q, k, v, g, beta, state = (t.to(MPS) for t in inputs(B, T, seed=1))
    full = metal.gated_delta_rule_outputs(q, k, v, g, beta, state.clone())
    steps = torch.tensor([0, 3, T], device=MPS)
    advanced = state.clone()
    metal.gated_delta_rule_advance_inplace(k, v, g, beta, advanced, steps)
    for b, j in enumerate(steps.tolist()):
        if j == 0 and not torch.equal(advanced[b], state[b]):
            failures.append("advancing by 0 steps changed the state")
        if 0 < j < T:
            rest = metal.gated_delta_rule_outputs(*(t[b:b + 1, j:].contiguous() for t in (q, k, v, g, beta)), advanced[b:b + 1].clone())
            if not torch.equal(rest[0], full[b, j:]):
                failures.append("outputs resumed from an advanced state differ from one uninterrupted run")
    clamped, exact_T = state.clone(), state.clone()
    metal.gated_delta_rule_advance_inplace(k, v, g, beta, clamped, torch.tensor([-5, T + 9, 0], device=MPS))
    metal.gated_delta_rule_advance_inplace(k, v, g, beta, exact_T, torch.tensor([0, T, 0], device=MPS))
    if not (torch.equal(clamped[0], state[0]) and torch.equal(clamped[1], exact_T[1])):
        failures.append("steps outside [0, T] are not clamped")
    padded = state.clone()
    metal.gated_delta_rule_outputs(q, k, v, torch.zeros_like(g), torch.zeros_like(beta), padded, advance_state=True)
    if not torch.equal(padded, state):
        failures.append("steps with g = 0 and beta = 0 changed the state")
    try:
        metal.gated_delta_rule_outputs(q.half(), k, v, g, beta, state.clone())
        failures.append("fp16 q was accepted")
    except ValueError:
        pass
    return failures


def eager_then_metal(fn):
    set_kernels(metal=False)
    try:
        eager = fn()
    finally:
        set_kernels(metal=True)
    return eager, fn()


@torch.no_grad()
def norms() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(2)
    for shape, scale in (((1, 16, 4096), 3.0), ((4, 16, 4096), 3.0), ((1, 180, 4096), 3.0), ((1, 16, 16, 256), 3.0), ((3, 7, 4, 256), 3.0),
                         ((2, 5, 4096), 1e-3), ((2, 5, 256), 1e-3)):
        norm = RMSNorm(shape[-1]).to(MPS, torch.bfloat16)
        norm.weight.copy_(torch.randn(shape[-1], generator=gen) * 0.1)
        x = (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16).to(MPS)
        eager, out = eager_then_metal(lambda: norm(x))
        equal = (out == eager).float().mean().item()
        worst = ((out.float() - eager.float()).abs() / eager.float().abs().clamp(min=1e-30)).max().item()
        print(f"RMSNorm {shape} x~{scale}: bitwise equal to eager {equal:.5f}, worst relative diff {worst:.2e}")
        if equal < 0.9999 or worst > torch.finfo(torch.bfloat16).eps:
            failures.append(f"RMSNorm {shape} x~{scale} differs from eager by more than one bf16 rounding")
    return failures


@torch.no_grad()
def rotary() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(4)
    for B, T, H, D in ((1, 12, 32, 128), (1, 16, 16, 256), (4, 9, 32, 128)):
        x = torch.randn(B, T, H, D, generator=gen).to(torch.bfloat16).to(MPS)
        positions = (torch.arange(T)[None] + torch.randint(0, 8000, (B, 1), generator=gen)).float()[..., None]
        freqs = positions * torch.logspace(0, -7, 32)
        cos, sin = freqs.cos().to(MPS), freqs.sin().to(MPS)
        eager, out = eager_then_metal(lambda: apply_rotary(x, cos, sin))
        if not torch.equal(out, eager):
            failures.append(f"apply_rotary {(B, T, H, D)} on Metal is not bitwise equal to eager")
    return failures


def exact_attention(q, k_cache, v_cache, mask, scale):
    B, Q, H, D = q.shape
    L, HKV = mask.shape[-1], k_cache.shape[2]
    k = k_cache[:B, :L].double().repeat_interleave(H // HKV, dim=2)
    v = v_cache[:B, :L].double().repeat_interleave(H // HKV, dim=2)
    scores = torch.einsum("bqhd,blhd->bhql", q.double(), k) * scale
    weights = torch.softmax(scores.masked_fill(~mask[:, None], float("-inf")), dim=-1).nan_to_num(0.0)
    return torch.einsum("bhql,blhd->bqhd", weights, v)


def attention() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(3)
    cases = [(kernel, 1, *case) for kernel in (metal.cached_attention,) for case in
             ((32, 32, 128, 12, 524, 1), (32, 32, 128, 12, 524, 8), (16, 4, 256, 16, 524, 8), (32, 32, 128, 12, 2572, 1), (16, 4, 256, 16, 2572, 8))]
    cases += [(metal.multi_query_attention, 3, *case) for case in ((32, 32, 128, 12, 524, 8), (16, 4, 256, 16, 700, 8), (32, 32, 128, 12, 2572, 1))]
    for kernel, B, H, HKV, D, Q, L, peak in cases:
        k_cache, v_cache = (torch.randn(B + 1, L + 40, HKV, D, generator=gen).to(torch.bfloat16) for _ in range(2))
        q = (torch.randn(B, Q, H, D, generator=gen) * peak).to(torch.bfloat16)
        full_mask = torch.rand(B, 1, 4 + Q, L, generator=gen) < 0.9
        full_mask[..., 0] = True
        full_mask[0, 0, 4 + 1] = False
        mask = full_mask[:, :, 4:][:, 0]
        exact = exact_attention(q, k_cache, v_cache, mask, D ** -0.5)
        on = lambda t: t.to(MPS)
        out = kernel(on(q), on(k_cache), on(v_cache), on(full_mask)[:, :, 4:][:, 0], D ** -0.5).cpu()
        live = mask.any(-1)
        sdpa = torch.nn.functional.scaled_dot_product_attention(
            on(q).transpose(1, 2), on(k_cache[:B, :L]).transpose(1, 2), on(v_cache[:B, :L]).transpose(1, 2), attn_mask=on(mask[:, None]),
            scale=D ** -0.5, enable_gqa=True).transpose(1, 2).cpu()
        rounded = exact.float().to(torch.bfloat16)
        metal_off, sdpa_off = ((t != rounded)[live].float().mean().item() for t in (out, sdpa))
        name = f"{kernel.__name__} B={B} H={H} HKV={HKV} D={D} L={L} peak x{peak}"
        print(f"{name}: not correctly rounded metal {metal_off:.5f} sdpa {sdpa_off:.5f}")
        if metal_off > 1.5 * sdpa_off + 1e-3:
            failures.append(f"{name} rounds worse than SDPA")
        if not torch.equal(out[~live], torch.zeros_like(out[~live])):
            failures.append(f"{name} gives a nonzero output for a query with no allowed position")
    return failures


def main() -> None:
    failures = exactness() + norms() + rotary() + attention()
    for B, T in ((1, 4), (3, 160), (1, 2048)):
        failures += accuracy(B, T)
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
