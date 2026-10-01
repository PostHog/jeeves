from __future__ import annotations

import sys

import torch

from model.model import (apply_rotary, conv_step, decode_attention, delta_gates, gated_delta_rule_advance_inplace, rotate_into_cache,
                         set_kernels)

CUDA = torch.device("cuda")
BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


def eager_then_triton(fn):
    set_kernels(triton_inference=False)
    try:
        eager = fn()
    finally:
        set_kernels(triton_inference=True)
    return eager, fn()


def same_bits(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x.isnan(), y.isnan()):
        return False
    numbers = ~x.isnan()
    return torch.equal(x.view(BITS[x.dtype])[numbers], y.view(BITS[y.dtype])[numbers])


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
        if not same_bits(out, eager):
            failures.append(f"apply_rotary {(B, T, H, D)} with the Triton kernel is not bitwise equal to eager")
        shared_eager, shared_out = eager_then_triton(lambda: apply_rotary(x, cos[0], sin[0]))
        if not same_bits(shared_out, shared_eager):
            failures.append(f"apply_rotary {(B, T, H, D)} with 2-D cos changed with the Triton kernel on")
    x = torch.randn(1, 2, 4, 128, generator=gen).to(torch.bfloat16).to(CUDA)
    x[0, 0, 0, 5] = float("nan")
    freqs = torch.arange(2).float()[None, :, None] * torch.logspace(0, -7, 32)
    eager, out = eager_then_triton(lambda: apply_rotary(x, freqs.cos().to(CUDA), freqs.sin().to(CUDA)))
    if not same_bits(out, eager):
        failures.append("apply_rotary with the Triton kernel does not propagate NaN like eager")
    return failures


@torch.no_grad()
def delta_advance() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(5)
    for B, T, H, HV, D in ((4, 4, 32, 32, 128), (2, 4, 16, 32, 128), (3, 1, 32, 32, 128), (4, 8, 32, 32, 128)):
        k = torch.randn(B, T, H, D, generator=gen).to(torch.bfloat16).to(CUDA)
        v = torch.randn(B, T, HV, D, generator=gen).to(torch.bfloat16).to(CUDA)
        g = -torch.rand(B, T, HV, generator=gen).to(CUDA)
        beta = torch.rand(B, T, HV, generator=gen).to(torch.bfloat16).to(CUDA)
        state = torch.randn(B, HV, D, D, generator=gen).to(CUDA)
        for steps in (torch.tensor([T, 0, 1, T - 1])[:B].to(CUDA), torch.tensor([-5, T + 9, 0, T])[:B].to(CUDA)):
            def advanced() -> torch.Tensor:
                advanced_state = state.clone()
                gated_delta_rule_advance_inplace(k, v, g, beta, advanced_state, steps)
                return advanced_state
            eager, out = eager_then_triton(advanced)
            if not same_bits(out, eager):
                failures.append(f"gated_delta_rule_advance_inplace {(B, T, H, HV, D)} with steps {steps.tolist()} and the Triton kernel is "
                                f"not bitwise equal to fla's masked steps")
            if not same_bits(out[steps <= 0], state[steps <= 0]):
                failures.append(f"gated_delta_rule_advance_inplace {(B, T, H, HV, D)} changed a row that advances 0 steps")
    return failures


@torch.no_grad()
def gates() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(6)
    HV = 32
    merged = (torch.randn(3, 5, 64 + 2 * HV, generator=gen) * 12).to(torch.bfloat16).to(CUDA)
    b, a = merged[..., 64:64 + HV], merged[..., 64 + HV:]
    # Softplus of about -103 to -87 is subnormal, where a flush to zero would differ from PyTorch.
    a[0, 0] = torch.linspace(-110, -80, HV)
    b[0, 0] = torch.linspace(-100, 100, HV)
    a[0, 1, :4] = torch.tensor([float("inf"), float("-inf"), 30.0, -30.0])
    b[0, 1, :2] = torch.tensor([float("inf"), float("-inf")])
    a[1, 0, 0] = b[1, 0, 1] = float("nan")
    rate = -torch.rand(HV, generator=gen).exp().to(CUDA)
    valid = (torch.rand(3, 5, generator=gen) < 0.7).to(CUDA)
    for dt_bias in (torch.randn(HV, generator=gen).to(CUDA), torch.randn(HV, generator=gen).to(torch.bfloat16).to(CUDA)):
        for rows_valid in (None, valid):
            eager, out = eager_then_triton(lambda: delta_gates(b, a, dt_bias, rate, rows_valid))
            for name, x, y in zip(("beta", "g"), out, eager):
                if not same_bits(x, y):
                    failures.append(f"delta_gates {name} (dt_bias {dt_bias.dtype}, valid {rows_valid is not None}) with the Triton kernel is "
                                    f"not bitwise equal to eager")
    return failures


@torch.no_grad()
def attention() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(7)
    K, J = 4, 3
    M = K * J
    block_of_mask = torch.arange(K).repeat_interleave(J)
    # Like the engine's decode masks: row t < K sees the cache up to n + t, and a mask row of block j sees the cache up to n + j and its block's tail keys, so whole slices of keys are masked.
    limit_offset = torch.cat((torch.arange(K), block_of_mask))
    allow_tail = torch.cat((torch.full((K,), -1), block_of_mask))[:, None] == block_of_mask[None, :]
    for B, first_row, H, Hkv, D, Lw in ((1, 0, 16, 4, 256, 512), (3, 0, 16, 4, 256, 1024), (2, K, 32, 32, 128, 2048), (4, K, 32, 32, 128, 512)):
        L, R = Lw + M, K + M - first_row
        q = torch.randn(B, R, H, D, generator=gen).to(torch.bfloat16).to(CUDA)
        k_cache, v_cache = (torch.randn(8, L + 1, Hkv, D, generator=gen).to(torch.bfloat16).to(CUDA) for _ in range(2))
        n = torch.randint(0, Lw - K, (B,), generator=gen)
        n[0] = 0
        allow_cache = torch.arange(Lw)[None, None] <= (n[:, None] + limit_offset[None])[..., None]
        allowed = torch.cat((allow_cache, allow_tail[None].expand(B, -1, -1)), dim=2)[:, None].to(CUDA)
        padded = torch.full((B, 1, K + M, L + 4), float("-inf"), dtype=torch.bfloat16, device=CUDA)[..., :L]
        mask = padded.masked_fill_(allowed, 0.0)[:, :, first_row:]
        eager, out = eager_then_triton(lambda: decode_attention(q, k_cache, v_cache, mask, D ** -0.5))
        keys, values = (t[:B, :L].float().repeat_interleave(H // Hkv, dim=2) for t in (k_cache, v_cache))
        scores = torch.einsum("brhd,blhd->bhrl", q.float(), keys) * D ** -0.5 + mask.float()
        reference = torch.einsum("bhrl,blhd->brhd", scores.softmax(-1), values)
        error, eager_error = (out.float() - reference).abs(), (eager.float() - reference).abs()
        if out.isnan().any() or error.mean() > 1.25 * eager_error.mean() or error.max() > 2 * eager_error.max():
            failures.append(f"decode_attention {(B, R, H, Hkv, D, L)}: the Triton kernel's error against float32 (mean {error.mean():.2e}, "
                            f"max {error.max():.2e}) is above SDPA's (mean {eager_error.mean():.2e}, max {eager_error.max():.2e})")
    return failures


@torch.no_grad()
def cache_writes() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(8)
    H, D, n_freqs, slots, tail_start = 32, 128, 32, 700, 512
    for B, K in ((1, 4), (4, 4), (2, 8)):
        M = K * (K - 1)
        k, v = (torch.randn(B, K, H, D, generator=gen).to(torch.bfloat16).to(CUDA) for _ in range(2))
        merged = torch.randn(B, M, 3 * H * D + 8, generator=gen).to(torch.bfloat16).to(CUDA)
        qm, km, vm = (merged[..., i * H * D:(i + 1) * H * D].view(B, M, H, D) for i in range(3))
        positions = (torch.randint(0, tail_start - K, (B, 1), generator=gen) + torch.arange(K)[None]).to(CUDA)
        freqs = torch.rand(B, K + M, 1, generator=gen) * 4000 * torch.logspace(0, -7, n_freqs)
        cos, sin = freqs.cos().to(CUDA), freqs.sin().to(CUDA)
        caches = [torch.randn(8, slots, H, D, generator=gen).to(torch.bfloat16).to(CUDA) for _ in range(2)]
        in_order = torch.arange(B, device=CUDA)[:, None].expand(B, K)
        shuffled = torch.randperm(8, generator=gen)[:B].to(CUDA)[:, None].expand(B, K)
        for rows, cos_rows in ((in_order, B), (shuffled, 1)):
            def written() -> tuple[torch.Tensor, ...]:
                k_cache, v_cache = (c.clone() for c in caches)
                c, s = cos[:cos_rows], sin[:cos_rows]
                q = rotate_into_cache(k, v, qm, km, vm, c[:, :K], s[:, :K], c[:, K:], s[:, K:], k_cache, v_cache, rows, positions, tail_start)
                return q, k_cache, v_cache
            eager, out = eager_then_triton(written)
            for name, x, y in zip(("queries", "key cache", "value cache"), out, eager):
                if not same_bits(x, y):
                    failures.append(f"rotate_into_cache {(B, K)} with cache rows {rows[:, 0].tolist()} and cos for {cos_rows} rows: the Triton "
                                    f"kernel's {name} is not bitwise equal to eager")
    return failures


@torch.no_grad()
def conv_steps() -> list[str]:
    failures = []
    gen = torch.Generator().manual_seed(9)
    key_dim, value_dim, head_dim, taps = 2048, 4096, 128, 4
    C = 2 * key_dim + value_dim
    for B, T in ((1, 4), (4, 4), (2, 8)):
        merged = (torch.randn(B, T, C + 64, generator=gen) * 2).to(torch.bfloat16).to(CUDA)
        state = torch.randn(B, C, taps, generator=gen).to(torch.bfloat16).to(CUDA)
        weight = (torch.randn(C, 1, taps, generator=gen) * 0.5).to(torch.bfloat16).to(CUDA)
        eager, out = eager_then_triton(lambda: conv_step(state, merged[..., :C], weight, key_dim, value_dim, head_dim, head_dim))
        for name, x, y in zip(("ext", "q", "k", "v"), out, eager):
            if not same_bits(x, y):
                failures.append(f"conv_step {(B, T)}: the Triton kernel's {name} is not bitwise equal to eager")
        # NaN in ext_out shows any output that reads ext before every thread has written it.
        stacked = torch.zeros(3, B, C, taps + T, dtype=torch.bfloat16, device=CUDA)
        stacked[1] = float("nan")
        out = conv_step(state, merged[..., :C], weight, key_dim, value_dim, head_dim, head_dim, ext_out=stacked[1])
        if out[0].data_ptr() != stacked[1].data_ptr() or not all(same_bits(x, y) for x, y in zip(out, eager)) or stacked[0].any() or stacked[2].any():
            failures.append(f"conv_step {(B, T)}: with ext_out, the Triton kernel did not write ext there alone, or its outputs changed")
    return failures


def main() -> None:
    enabled = set_kernels(triton_inference=True, gdn=True)
    if not (enabled["triton_inference"] and enabled["gdn"]):
        print(f"the Triton and fla kernels are not available, so nothing can be checked: {enabled}")
        sys.exit(1)
    failures = rotary() + delta_advance() + gates() + attention() + cache_writes() + conv_steps()
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
