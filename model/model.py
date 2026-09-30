from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint

from . import metal
from .config import LINEAR, Qwen3_5_9BConfig
from .lora import LoRAAdapter, LoRAConfig, LoRALinear, inject_lora, mark_only_lora_trainable

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdr
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as _fla_recurrent_gdr
except ImportError:
    _fla_chunk_gdr = _fla_recurrent_gdr = None
try:
    import triton
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as _fla_recurrent_kernel
except ImportError:
    triton = _fla_recurrent_kernel = None
try:
    from . import triton_kernels
except ImportError:
    triton_kernels = None
try:
    from fla.modules.layernorm import rms_norm as _fla_rms_norm
except ImportError:
    _fla_rms_norm = None
try:
    from fla.modules.fused_norm_gate import rms_norm_gated as _fla_rms_norm_gated
except ImportError:
    _fla_rms_norm_gated = None
try:
    from fla.modules.activations import swiglu as _fla_swiglu
except ImportError:
    _fla_swiglu = None
try:
    from fla.modules.rotary import rotary_embedding as _fla_rotary
except ImportError:
    _fla_rotary = None
try:
    from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss as _FlaFusedLCE
except ImportError:
    _FlaFusedLCE = None
try:
    from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d
    from fla.modules.convolution import causal_conv1d_update as _fla_causal_conv1d_update
except ImportError:
    _fla_causal_conv1d = _fla_causal_conv1d_update = None
try:
    from flash_attn import flash_attn_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None

_AVAILABLE = {
    "gdn": _fla_chunk_gdr is not None and _fla_recurrent_gdr is not None,
    "rms_norm": _fla_rms_norm is not None,
    "rms_norm_gated": _fla_rms_norm_gated is not None,
    "swiglu": _fla_swiglu is not None,
    "rotary": _fla_rotary is not None,
    "fused_lce": _FlaFusedLCE is not None,
    "causal_conv1d": _fla_causal_conv1d is not None and _fla_causal_conv1d_update is not None,
    "flash_attn": _flash_attn_func is not None,
    "metal": torch.backends.mps.is_available(),
    "triton_inference": triton_kernels is not None,
}
_ENABLED_BY_ENV = os.environ.get("QWEN35_KERNELS", "1") not in ("0", "false", "False")
KERNELS = {k: v and _ENABLED_BY_ENV for k, v in _AVAILABLE.items()}


def set_kernels(**flags: bool) -> dict[str, bool]:
    if "all" in flags:
        on = flags.pop("all")
        for k in KERNELS:
            KERNELS[k] = on and _AVAILABLE[k]
    for k, on in flags.items():
        if k not in KERNELS:
            raise KeyError(f"unknown kernel {k!r}; choose from {sorted(KERNELS)}")
        KERNELS[k] = on and _AVAILABLE[k]
    return dict(KERNELS)


def _on(name: str, x: torch.Tensor) -> bool:
    return KERNELS[name] and x.is_cuda


def _needs_grad(*xs: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(x.requires_grad for x in xs)


def metal_enabled() -> bool:
    return KERNELS["metal"] and not torch.compiler.is_compiling()


def _on_metal(*xs: torch.Tensor) -> bool:
    return metal_enabled() and all(x.is_mps and x.dtype == torch.bfloat16 for x in xs) and not _needs_grad(*xs)


def _on_triton_inference(x: torch.Tensor, *others: torch.Tensor) -> bool:
    return _on("triton_inference", x) and x.dtype == torch.bfloat16 and not torch.compiler.is_compiling() and not _needs_grad(x, *others)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _on("rms_norm", x):
            return _fla_rms_norm(x, 1.0 + self.weight.float(), None, eps=self.eps)
        if _on_metal(x, self.weight):
            return metal.rms_norm(x, self.weight, self.eps)
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).type_as(x)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class FrozenRMSNorm(RMSNorm):
    def __init__(self, norm: RMSNorm):
        super().__init__(norm.weight.shape[0], norm.eps)
        self.weight = norm.weight
        self.register_buffer("shifted_weight", 1.0 + norm.weight.detach().float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _on("rms_norm", x):
            return _fla_rms_norm(x, self.shifted_weight, None, eps=self.eps)
        return super().forward(x)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if _on("rms_norm_gated", x):
            return _fla_rms_norm_gated(x, z, self.weight, None, activation="swish", eps=self.eps)
        if x.shape == z.shape and _on_metal(x, z, self.weight):
            return metal.gated_rms_norm(x, z, self.weight, self.eps)
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.weight * xf.to(x.dtype)
        return (y * F.silu(z.float())).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: Qwen3_5_9BConfig):
        super().__init__()
        self.dim = cfg.rotary_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and _on_metal(x) and not _needs_grad(cos, sin):
        return metal.rotary(x, cos, sin)
    # Only for per-row positions (3-D cos), where the kernel replaces the eager path; with 2-D cos fla's rotary runs, which rounds differently.
    if x.dim() == 4 and cos.dim() == 3 and _on_triton_inference(x, cos, sin):
        return triton_kernels.rotary(x, cos, sin)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    if cos.dim() == 2 and _on("rotary", x):
        return _fla_rotary(x, cos, sin, interleaved=False)
    rd = 2 * cos.shape[-1]
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if _on("causal_conv1d", x):
        return _fla_causal_conv1d(x, weight, None, activation="silu")[0]
    K = weight.shape[-1]
    y = F.conv1d(x.transpose(1, 2), weight.unsqueeze(1), None, padding=K - 1, groups=x.shape[-1])
    return F.silu(y[..., : x.shape[1]]).transpose(1, 2)


def causal_conv1d_step(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if _on("causal_conv1d", x) and not _needs_grad(x, weight):
        return _fla_causal_conv1d_update(x, state, None, weight, None, "silu")[0]
    K = weight.shape[-1]
    window = torch.cat((state, x.to(state.dtype)[..., None]), dim=-1)[..., -K:]
    state.copy_(window)
    y = (window * weight.unsqueeze(0)).sum(-1)
    return F.silu(y).to(x.dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None, output_final_state=False):
    dtype = q.dtype
    q, k = _l2norm(q.float()), _l2norm(k.float())
    v, g, beta = v.float(), g.float(), beta.float()
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    q = q * Dk**-0.5
    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=q.device) if initial_state is None \
        else initial_state.to(torch.float32)
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]
        retrieved = torch.einsum("bhk,bhkv->bhv", k_t, S)
        delta = beta[:, t, :, None] * (v_t - retrieved)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        outs.append(torch.einsum("bhk,bhkv->bhv", q_t, S))
    out = torch.stack(outs, dim=1)
    return out.to(dtype), (S if output_final_state else None)


def torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, initial_state=None, output_final_state=False):
    dtype = q.dtype
    q, k = _l2norm(q.float()), _l2norm(k.float())
    v, g, beta = v.float(), g.float(), beta.float()
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (-T) % C
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, 0, 0, pad)) for x in (g, beta))
    N = (T + pad) // C
    q = (q * Dk**-0.5).transpose(1, 2).reshape(B, H, N, C, Dk)
    k = k.transpose(1, 2).reshape(B, H, N, C, Dk)
    v = v.transpose(1, 2).reshape(B, H, N, C, Dv)
    g = g.transpose(1, 2).reshape(B, H, N, C)
    beta = beta.transpose(1, 2).reshape(B, H, N, C)

    gc = g.cumsum(-1)
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=-1)
    decay = (gc[..., :, None] - gc[..., None, :]).masked_fill(~lower, 0).exp().masked_fill(~lower, 0)

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    L = (k_beta @ k.transpose(-1, -2) * decay).masked_fill(~strict, 0)
    L = L + torch.eye(C, dtype=L.dtype, device=L.device)
    u = torch.linalg.solve_triangular(L, v_beta, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(L, k_beta * gc.exp()[..., None], upper=False, unitriangular=True)

    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=q.device) if initial_state is None \
        else initial_state.to(torch.float32)
    outs = []
    for n in range(N):
        q_n, k_n, gc_n = q[:, :, n], k[:, :, n], gc[:, :, n]
        v_new = u[:, :, n] - w[:, :, n] @ S
        intra = (q_n @ k_n.transpose(-1, -2) * decay[:, :, n]) @ v_new
        inter = (q_n * gc_n.exp()[..., None]) @ S
        outs.append(inter + intra)
        g_last = gc_n[..., -1]
        S = S * g_last.exp()[..., None, None] + \
            (k_n * (g_last[..., None] - gc_n).exp()[..., None]).transpose(-1, -2) @ v_new

    out = torch.stack(outs, dim=2).reshape(B, H, N * C, Dv)[:, :, :T].transpose(1, 2)
    return out.to(dtype), (S if output_final_state else None)


def gated_delta_rule_chunk(q, k, v, g, beta, initial_state=None, output_final_state=False):
    if _on("gdn", q):
        return _fla_chunk_gdr(q, k, v, g=g, beta=beta, initial_state=initial_state,
                              output_final_state=output_final_state, use_qk_l2norm_in_kernel=True)
    return torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=initial_state,
                                        output_final_state=output_final_state)


def gated_delta_rule_step_inplace(q, k, v, g, beta, state):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    o = torch.empty_like(v)
    _fla_recurrent_kernel[(triton.cdiv(V, BV), B * HV)](
        q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None, o=o, h0=state, ht=state, cu_seqlens=None,
        scale=K**-0.5, T=T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, IS_BETA_HEADWISE=True, USE_QK_L2NORM_IN_KERNEL=True,
        APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False, num_warps=1, num_stages=3)
    return o


def gated_delta_rule_step(q, k, v, g, beta, initial_state=None, output_final_state=True):
    if _on("gdn", q) and not _needs_grad(q, k, v, g, beta):
        return _fla_recurrent_gdr(q, k, v, g=g, beta=beta, initial_state=initial_state,
                                  output_final_state=output_final_state, use_qk_l2norm_in_kernel=True)
    return torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=initial_state,
                                            output_final_state=output_final_state)


class Cache:
    def __init__(self, num_layers: int):
        self.kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        self.conv: list[torch.Tensor | None] = [None] * num_layers
        self.rec: list[torch.Tensor | None] = [None] * num_layers
        self.seen_tokens = 0

    def has_state(self, layer: int) -> bool:
        return self.kv[layer] is not None or self.rec[layer] is not None

    def update_kv(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kv[layer] is not None:
            k = torch.cat((self.kv[layer][0], k), dim=1)
            v = torch.cat((self.kv[layer][1], v), dim=1)
        self.kv[layer] = (k, v)
        return k, v

    def reorder(self, idx: torch.Tensor) -> None:
        for i in range(len(self.kv)):
            if self.kv[i] is not None:
                self.kv[i] = (self.kv[i][0][idx], self.kv[i][1][idx])
            if self.conv[i] is not None:
                self.conv[i] = self.conv[i][idx]
            if self.rec[i] is not None:
                self.rec[i] = self.rec[i][idx]


class StaticCache:
    def __init__(self, cfg: Qwen3_5_9BConfig, batch: int, max_len: int, device: torch.device, dtype: torch.dtype):
        self.max_len = max_len
        self.prefilled = False
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.conv: dict[int, torch.Tensor] = {}
        self.rec: dict[int, torch.Tensor] = {}
        for i, kind in enumerate(cfg.layer_types):
            if kind == LINEAR:
                self.conv[i] = torch.zeros(batch, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim, device=device, dtype=dtype)
                self.rec[i] = torch.zeros(batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim,
                                          device=device, dtype=torch.float32)
            else:
                self.k[i] = torch.zeros(batch, max_len, cfg.num_key_value_heads, cfg.head_dim, device=device, dtype=dtype)
                self.v[i] = torch.zeros(batch, max_len, cfg.num_key_value_heads, cfg.head_dim, device=device, dtype=dtype)
        self.valid = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
        self.write_idx = torch.zeros(1, dtype=torch.long, device=device)
        self.window = max_len

    def decoding(self, T: int) -> bool:
        return self.prefilled and T == 1

    def compact(self, rows: torch.Tensor) -> None:
        for d in (self.k, self.v, self.conv, self.rec):
            for i in d:
                d[i] = d[i].index_select(0, rows).contiguous()
        self.valid = self.valid.index_select(0, rows).contiguous()

    def finish_prefill(self, T: int, attention_mask: torch.Tensor | None) -> None:
        self.valid[:, :T] = True if attention_mask is None else attention_mask.bool()
        self.write_idx.fill_(T)
        self.prefilled = True

    def begin_step(self) -> None:
        self.valid.index_fill_(1, self.write_idx, True)

    def advance(self) -> None:
        self.write_idx += 1


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen3_5_9BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_k_heads, self.num_v_heads = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.head_k_dim, self.head_v_dim = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = cfg.linear_key_dim, cfg.linear_value_dim
        self.conv_dim, self.conv_kernel = cfg.linear_conv_dim, cfg.linear_conv_kernel_dim

        hs = cfg.hidden_size
        self.in_proj_qkv = nn.Linear(hs, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hs, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hs, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hs, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.conv_kernel, groups=self.conv_dim,
                                padding=self.conv_kernel - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = GatedRMSNorm(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hs, bias=False)

    def forward(self, x: torch.Tensor, cache: Cache | None = None,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        if padding_mask is not None:
            x = (x * padding_mask[:, :, None]).to(x.dtype)

        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).view(B, T, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        conv_w = self.conv1d.weight.squeeze(1)

        if isinstance(cache, StaticCache):
            if cache.decoding(T):
                qkv = causal_conv1d_step(qkv[:, 0], cache.conv[self.layer_idx], conv_w)[:, None]
            else:
                cache.conv[self.layer_idx].copy_(F.pad(qkv.transpose(1, 2), (self.conv_kernel - T, 0)))
                qkv = causal_conv1d(qkv, conv_w)
            cached = cache.prefilled
        else:
            cached = cache is not None and cache.has_state(self.layer_idx)
            if cached and T == 1:
                qkv = causal_conv1d_step(qkv[:, 0], cache.conv[self.layer_idx], conv_w)[:, None]
            else:
                if cached:
                    qkv = torch.cat((cache.conv[self.layer_idx].transpose(1, 2), qkv), dim=1)
                if cache is not None:
                    cache.conv[self.layer_idx] = F.pad(qkv.transpose(1, 2), (self.conv_kernel - qkv.shape[1], 0)).contiguous()
                qkv = causal_conv1d(qkv, conv_w)[:, -T:]

        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads != self.num_k_heads:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if isinstance(cache, StaticCache):
            if cache.decoding(T):
                if _on("gdn", q) and _fla_recurrent_kernel is not None and not _needs_grad(q, k, v, g, beta):
                    o = gated_delta_rule_step_inplace(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(),
                                                      beta.contiguous(), cache.rec[self.layer_idx])
                else:
                    o, state = gated_delta_rule_step(q, k, v, g, beta, initial_state=cache.rec[self.layer_idx], output_final_state=True)
                    cache.rec[self.layer_idx].copy_(state)
            else:
                o, state = gated_delta_rule_chunk(q, k, v, g, beta, initial_state=None, output_final_state=True)
                cache.rec[self.layer_idx].copy_(state)
        else:
            state = cache.rec[self.layer_idx] if cached else None
            if cached and T == 1:
                o, state = gated_delta_rule_step(q, k, v, g, beta, initial_state=state, output_final_state=True)
            else:
                o, state = gated_delta_rule_chunk(q, k, v, g, beta, initial_state=state,
                                                  output_final_state=cache is not None)
            if cache is not None:
                cache.rec[self.layer_idx] = state

        o = self.norm(o.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(o.view(B, T, self.value_dim))


class GatedAttention(nn.Module):
    def __init__(self, cfg: Qwen3_5_9BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.scaling = self.head_dim**-0.5
        hs, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(hs, self.num_heads * hd * 2, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(hs, self.num_kv_heads * hd, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(hs, self.num_kv_heads * hd, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * hd, hs, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor | None = None, cache: Cache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        q, gate = self.q_proj(x).view(B, T, H, 2 * D).chunk(2, dim=-1)
        gate = gate.reshape(B, T, H * D)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(B, T, Hkv, D))
        v = self.v_proj(x).view(B, T, Hkv, D)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if isinstance(cache, StaticCache):
            if cache.decoding(T):
                cache.k[self.layer_idx].index_copy_(1, cache.write_idx, k)
                cache.v[self.layer_idx].index_copy_(1, cache.write_idx, v)
                k, v = cache.k[self.layer_idx][:, :cache.window], cache.v[self.layer_idx][:, :cache.window]
                attn_mask = cache.valid[:, None, None, :cache.window]
            else:
                cache.k[self.layer_idx][:, :T].copy_(k)
                cache.v[self.layer_idx][:, :T].copy_(v)
        elif cache is not None:
            k, v = cache.update_kv(self.layer_idx, k, v)
        Tk = k.shape[1]
        causal = T > 1 and attn_mask is None

        if attn_mask is None and _on("flash_attn", q) and q.dtype in (torch.float16, torch.bfloat16):
            o = _flash_attn_func(q, k, v, softmax_scale=self.scaling, causal=causal)
        else:
            qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
            if attn_mask is None and T > 1 and T != Tk:
                attn_mask = torch.ones(T, Tk, dtype=torch.bool, device=q.device).tril(Tk - T)
                causal = False
            o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask, is_causal=causal,
                                               scale=self.scaling, enable_gqa=True)
            o = o.transpose(1, 2)
        o = o.reshape(B, T, H * D) * torch.sigmoid(gate)
        return self.o_proj(o)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Qwen3_5_9BConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(x), self.up_proj(x)
        h = _fla_swiglu(gate, up) if _on("swiglu", x) else F.silu(gate) * up
        return self.down_proj(h)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3_5_9BConfig, layer_idx: int):
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        if self.layer_type == LINEAR:
            self.linear_attn = GatedDeltaNet(cfg, layer_idx)
        else:
            self.self_attn = GatedAttention(cfg, layer_idx)
        self.mlp = SwiGLU(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask, padding_mask, cache=None):
        h = self.input_layernorm(x)
        if self.layer_type == LINEAR:
            h = self.linear_attn(h, cache=cache, padding_mask=padding_mask)
        else:
            h = self.self_attn(h, cos, sin, attn_mask=attn_mask, cache=cache)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg: Qwen3_5_9BConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg)
        self.gradient_checkpointing = False
        self.compiled_layers: list | None = None

    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, cache: Cache | None = None,
                inputs_embeds: torch.Tensor | None = None, normed: bool = True) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        B, T, _ = x.shape
        static = isinstance(cache, StaticCache)
        past = cache.seen_tokens if cache is not None and not static else 0

        if position_ids is None:
            pos = torch.arange(past, past + T, device=x.device)
        else:
            pos = position_ids
            if pos.dim() == 2 and pos.shape[0] == 1:
                pos = pos[0]
        cos, sin = self.rotary_emb(pos)

        attn_mask = padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask.to(x.dtype)
            key_valid = attention_mask.bool()
            if past and key_valid.shape[1] == T:
                key_valid = torch.cat((torch.ones(B, past, dtype=torch.bool, device=x.device), key_valid), 1)
            Tk = key_valid.shape[1]
            causal = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(Tk - T)
            attn_mask = causal[None, None] & key_valid[:, None, None, :]
            if past:
                padding_mask = None

        if static and cache.decoding(T):
            cache.begin_step()
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled() and cache is None
        layers = self.compiled_layers if self.compiled_layers is not None and cache is None else self.layers
        for layer in layers:
            if use_ckpt:
                x = checkpoint(layer, x, cos, sin, attn_mask, padding_mask, use_reentrant=False)
            else:
                x = layer(x, cos, sin, attn_mask, padding_mask, cache=cache)
        if static:
            if cache.decoding(T):
                cache.advance()
            else:
                cache.finish_prefill(T, attention_mask)
        elif cache is not None:
            cache.seen_tokens += T
        return self.norm(x) if normed else x


@dataclass
class CausalLMOutput:
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    cache: Cache | None = None


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3_5_9BConfig | None = None, lora: LoRAConfig | None = None):
        super().__init__()
        self.cfg = cfg or Qwen3_5_9BConfig()
        self.model = Qwen3_5Model(self.cfg)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.lora: LoRAAdapter | None = None
        self.freeze()
        if lora is not None:
            self.add_lora(lora)

    def freeze(self) -> "Qwen3_5ForCausalLM":
        for p in self.parameters():
            p.requires_grad_(False)
        if self.lora is not None:
            self.lora.requires_grad_(True)
        return self.eval()

    def add_lora(self, cfg: LoRAConfig | None = None) -> LoRAAdapter:
        if self.lora is not None:
            raise RuntimeError("LoRA already attached; call remove_lora() first")
        cfg = cfg or LoRAConfig()
        modules = inject_lora(self, cfg)
        mark_only_lora_trainable(self)
        self.lora = LoRAAdapter(modules, cfg)
        return self.lora

    def remove_lora(self, merge: bool = False) -> None:
        if self.lora is None:
            return
        for name, wrapped in self.lora.modules():
            if merge:
                wrapped.merge()
            parent_name, _, attr = name.rpartition(".")
            setattr(self.get_submodule(parent_name) if parent_name else self, attr, wrapped.base)
        self.lora = None
        self.freeze()

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen or (trainable_only and not p.requires_grad):
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def gradient_checkpointing_enable(self, on: bool = True) -> None:
        self.model.gradient_checkpointing = on

    def compile_layers(self, dynamic: bool = True, mode: str | None = None) -> None:
        self.model.compiled_layers = [torch.compile(layer, dynamic=dynamic, mode=mode) for layer in self.model.layers]

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                      position_ids: torch.Tensor | None = None, cache: Cache | None = None) -> torch.Tensor:
        return self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids, cache=cache)

    def token_logprobs(self, hidden: torch.Tensor, targets: torch.Tensor, chunk_size: int = 128,
                       temperature: float = 1.0, logit_bias: torch.Tensor | None = None) -> torch.Tensor:
        weight = self.lm_head.weight
        out = []
        for start in range(0, hidden.shape[1], chunk_size):
            h = hidden[:, start:start + chunk_size]
            t = targets[:, start:start + chunk_size]
            out.append(checkpoint(_chunk_logprobs, h, t, weight, temperature, logit_bias, use_reentrant=False) if torch.is_grad_enabled()
                       else _chunk_logprobs(h, t, weight, temperature, logit_bias))
        return torch.cat(out, dim=1)

    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, cache: Cache | None = None,
                inputs_embeds: torch.Tensor | None = None, labels: torch.Tensor | None = None,
                logits_to_keep: int = 0, return_logits: bool | None = None) -> CausalLMOutput:
        h = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids, cache=cache,
                       inputs_embeds=inputs_embeds)
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        if return_logits is None:
            return_logits = labels is None

        loss = logits = None
        if labels is not None:
            h_s, y = h[:, :-1], labels[:, 1:]
            if not return_logits and _on("fused_lce", h) and type(self.lm_head) is nn.Linear:
                loss = _FlaFusedLCE(ignore_index=-100)(h_s.reshape(-1, h_s.shape[-1]), y.reshape(-1),
                                                       self.lm_head.weight, self.lm_head.bias)
            else:
                logits = self.lm_head(h)
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                       ignore_index=-100)
        if return_logits and logits is None:
            logits = self.lm_head(h)
        return CausalLMOutput(logits=logits, loss=loss, cache=cache)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, attention_mask: torch.Tensor | None = None,
                 temperature: float = 0.0, top_p: float = 1.0, eos_token_id: int | None = None,
                 stop_on_eos: bool = True, sync_every: int = 8, logit_bias: torch.Tensor | None = None) -> torch.Tensor:
        eos = self.cfg.eos_token_id if eos_token_id is None else eos_token_id
        B = input_ids.shape[0]
        cache = Cache(self.cfg.num_hidden_layers)
        position_ids = None
        if attention_mask is not None:
            m = attention_mask.bool()
            if bool((m[:, :-1] & ~m[:, 1:]).any()) or not bool(m[:, -1].all()):
                raise ValueError("generate expects left padding: each attention_mask row must be 0...01...1")
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self(input_ids, attention_mask=attention_mask, position_ids=position_ids, cache=cache, logits_to_keep=1)
        next_pos = (position_ids[:, -1] if position_ids is not None
                    else torch.full((B,), input_ids.shape[1] - 1, device=input_ids.device))
        tokens = [input_ids]
        done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        for i in range(max_new_tokens):
            lg = out.logits[:, -1].float()
            if logit_bias is not None:
                lg = lg + logit_bias
            nxt = self._sample(lg, temperature, top_p)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            tokens.append(nxt[:, None])
            done |= nxt == eos
            if stop_on_eos and (i + 1) % sync_every == 0 and bool(done.all()):
                break
            next_pos = next_pos + 1
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype,
                                                                       device=attention_mask.device)), 1)
            out = self(nxt[:, None], attention_mask=attention_mask, position_ids=next_pos[:, None], cache=cache,
                       logits_to_keep=1)
        return torch.cat(tokens, dim=1)

    @torch.no_grad()
    def generate_graphed(self, input_ids: torch.Tensor, max_new_tokens: int = 64, attention_mask: torch.Tensor | None = None,
                         temperature: float = 1.0, eos_token_id: int | None = None, stop_on_eos: bool = True,
                         sync_every: int = 16, warmup_steps: int = 2, window_step: int = 512,
                         compact_below: float = 0.75, return_logprobs: bool = False,
                         capture_lock=None,
                         logit_bias: torch.Tensor | None = None):
        eos = self.cfg.eos_token_id if eos_token_id is None else eos_token_id
        B, T = input_ids.shape
        device = input_ids.device
        if attention_mask is not None:
            m = attention_mask.bool()
            if bool((m[:, :-1] & ~m[:, 1:]).any()) or not bool(m[:, -1].all()):
                raise ValueError("generate expects left padding: each attention_mask row must be 0...01...1")
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        else:
            position_ids = torch.arange(T, device=device)[None].expand(B, T)
        max_len = T + max_new_tokens
        cache = StaticCache(self.cfg, B, max_len, device, self.model.embed_tokens.weight.dtype)
        h = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids, cache=cache)
        logits = self.lm_head(h[:, -1]).float()
        if logit_bias is not None:
            logits = logits + logit_bias
        cache.window = min(max_len, -(-(T + 1) // window_step) * window_step)

        st = {"tok": torch.zeros(B, 1, dtype=torch.long, device=device), "pos": (position_ids[:, -1:] + 1).contiguous(),
              "done": torch.zeros(B, dtype=torch.bool, device=device), "rows": torch.arange(B, device=device),
              "lp": torch.zeros(B, dtype=torch.float32, device=device)}
        out = torch.full((B, max_new_tokens), eos, dtype=torch.long, device=device)
        out_lp = torch.zeros(B, max_new_tokens, dtype=torch.float32, device=device)
        inv_t = 1.0 / temperature if temperature > 0 else 1.0

        def sample(lg: torch.Tensor) -> torch.Tensor:
            if temperature <= 0:
                return lg.argmax(-1)
            return (lg * inv_t - torch.empty_like(lg).exponential_().log()).argmax(-1)

        def commit(lg: torch.Tensor, nxt: torch.Tensor) -> None:
            nxt = torch.where(st["done"], torch.full_like(nxt, eos), nxt)
            scaled = lg * inv_t
            st["lp"].copy_(scaled.gather(1, nxt[:, None]).squeeze(1) - torch.logsumexp(scaled, dim=-1))
            st["done"].logical_or_(nxt == eos)
            st["tok"].copy_(nxt[:, None])

        commit(logits, sample(logits))

        def step() -> None:
            hs = self.model(st["tok"], position_ids=st["pos"], cache=cache)
            lg = self.lm_head(hs[:, -1]).float()
            if logit_bias is not None:
                lg = lg + logit_bias
            commit(lg, sample(lg))
            st["pos"].add_(1)

        def capture() -> torch.cuda.CUDAGraph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                step()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            lock = capture_lock if capture_lock is not None else contextlib.nullcontext()
            with lock:
                torch.cuda.synchronize()
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    step()
                torch.cuda.synchronize()
            return g

        graph = None
        for i in range(max_new_tokens):
            out[st["rows"], i] = st["tok"][:, 0]
            out_lp[st["rows"], i] = st["lp"]
            if i == max_new_tokens - 1:
                break
            if T + i + 2 > cache.window:
                cache.window = min(max_len, cache.window + window_step)
                alive = (~st["done"]).nonzero().squeeze(1)
                if alive.numel() == 0 and stop_on_eos:
                    break
                if alive.numel() < compact_below * st["rows"].numel():
                    cache.compact(alive)
                    for key in ("tok", "pos", "done", "rows", "lp"):
                        st[key] = st[key].index_select(0, alive).contiguous()
                graph = None
            elif stop_on_eos and (i + 1) % sync_every == 0 and bool(st["done"].all()):
                break
            if i < warmup_steps:
                step()
            elif graph is None:
                graph = capture()
            else:
                graph.replay()
        full = torch.cat((input_ids, out), dim=1)
        return (full, out_lp) if return_logprobs else full

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sp, si = probs.sort(-1, descending=True)
            keep = (sp.cumsum(-1) - sp) < top_p
            sp = sp * keep
            probs = torch.zeros_like(probs).scatter(-1, si, sp / sp.sum(-1, keepdim=True))
        return torch.multinomial(probs, 1).squeeze(-1)

    @classmethod
    def from_pretrained(cls, path_or_repo: str = "Qwen/Qwen3.5-9B", cfg: Qwen3_5_9BConfig | None = None,
                        device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16,
                        lora: LoRAConfig | None = None, strict: bool = True) -> "Qwen3_5ForCausalLM":
        path = resolve_checkpoint(path_or_repo)
        cfg = cfg or Qwen3_5_9BConfig.from_json(path)
        with torch.device(device), _default_dtype(dtype):
            model = cls(cfg)
        load_qwen_weights(model, path, strict=strict)
        if lora is not None:
            model.add_lora(lora)
        return model.freeze()


def _chunk_logprobs(h: torch.Tensor, t: torch.Tensor, weight: torch.Tensor, temperature: float = 1.0,
                    logit_bias: torch.Tensor | None = None) -> torch.Tensor:
    logits = F.linear(h, weight).float()
    if temperature != 1.0:
        logits = logits / temperature
    if logit_bias is not None:
        logits = logits + logit_bias
    return logits.gather(-1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def resolve_checkpoint(path_or_repo: str) -> str:
    if os.path.isdir(path_or_repo):
        return path_or_repo
    return snapshot_download(path_or_repo, allow_patterns=["*.safetensors", "*.json"])


_SKIP_PREFIXES = ("model.visual.", "mtp.")
_RENAMES = (("model.language_model.", "model."),)


def map_checkpoint_key(key: str) -> str | None:
    if key.startswith(_SKIP_PREFIXES):
        return None
    for old, new in _RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


@torch.no_grad()
def load_qwen_weights(model: Qwen3_5ForCausalLM, path: str, strict: bool = True) -> dict[str, list[str]]:
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found under {path}")

    params = dict(model.named_parameters())
    params.update(model.named_buffers())
    loaded, unexpected = set(), []
    for shard in shards:
        with safe_open(os.path.join(path, shard), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                name = map_checkpoint_key(key)
                if name is None:
                    continue
                if name not in params:
                    unexpected.append(key)
                    continue
                dst = params[name]
                src = fh.get_tensor(key)
                if src.shape != dst.shape:
                    raise ValueError(f"{key}: checkpoint {tuple(src.shape)} vs model {tuple(dst.shape)}")
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype), non_blocking=True)
                loaded.add(name)

    if "lm_head.weight" not in loaded and "model.embed_tokens.weight" in loaded:
        model.lm_head.weight = model.model.embed_tokens.weight
        loaded.add("lm_head.weight")
    missing = sorted(k for k in params if k not in loaded and not k.endswith("inv_freq"))
    if strict and (missing or unexpected):
        raise RuntimeError(f"weight loading mismatch.\n  missing: {missing[:8]}\n  unexpected: {unexpected[:8]}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"missing": missing, "unexpected": unexpected}


__all__ = ["Qwen3_5ForCausalLM", "Qwen3_5Model", "Cache", "CausalLMOutput", "load_qwen_weights",
           "resolve_checkpoint", "set_kernels", "KERNELS", "LoRAConfig", "LoRAAdapter", "LoRALinear"]
