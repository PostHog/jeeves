from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.config import LINEAR
from model.model import Qwen3_5ForCausalLM, apply_rotary, causal_conv1d, gated_delta_rule_chunk


class _CachedCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
        return cached

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad.to(torch.float32), None


class OrthrusView(nn.Module):
    def __init__(self, base: Qwen3_5ForCausalLM, block: int = 8):
        super().__init__()
        self.base = base
        self.cfg = cfg = base.cfg
        self.block = block
        self.dtype = base.lm_head.weight.dtype
        for p in base.parameters():
            p.requires_grad_(False)
        hs = cfg.hidden_size
        self.mask_embed = nn.Parameter(base.model.embed_tokens.weight.float().mean(0).clone())
        self.attn_q = nn.ModuleDict()
        self.attn_k = nn.ModuleDict()
        self.attn_v = nn.ModuleDict()
        self.delta_q = nn.ModuleDict()
        self.delta_k = nn.ModuleDict()
        self.delta_v = nn.ModuleDict()
        for i, layer in enumerate(base.model.layers):
            key = str(i)
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                w = lin.in_proj_qkv.weight.float()
                rep = lin.num_v_heads // lin.num_k_heads
                wq = w[:lin.key_dim].view(lin.num_k_heads, lin.head_k_dim, hs).repeat_interleave(rep, dim=0).reshape(lin.value_dim, hs)
                wk = w[lin.key_dim:2 * lin.key_dim].view(lin.num_k_heads, lin.head_k_dim, hs).repeat_interleave(rep, dim=0).reshape(lin.value_dim, hs)
                wv = w[2 * lin.key_dim:2 * lin.key_dim + lin.value_dim]
                for src, store in ((wq, self.delta_q), (wk, self.delta_k), (wv, self.delta_v)):
                    proj = nn.Linear(hs, lin.value_dim, bias=False)
                    proj.weight.data.copy_(src)
                    store[key] = proj
            else:
                att = layer.self_attn
                for src, store in ((att.q_proj, self.attn_q), (att.k_proj, self.attn_k), (att.v_proj, self.attn_v)):
                    proj = nn.Linear(hs, src.out_features, bias=False)
                    proj.weight.data.copy_(src.weight.float())
                    store[key] = proj
        self.grad_checkpoint = True
        self.compiled_mask_layers: list | None = None
        self.weight_cache: dict[int, torch.Tensor] = {}

    def projections(self) -> list[nn.Linear]:
        return [p for d in (self.attn_q, self.attn_k, self.attn_v, self.delta_q, self.delta_k, self.delta_v) for p in d.values()]

    @torch.no_grad()
    def refresh_cache(self) -> None:
        for proj in self.projections():
            buf = self.weight_cache.get(id(proj))
            if buf is None:
                buf = torch.empty_like(proj.weight, dtype=self.dtype)
                self.weight_cache[id(proj)] = buf
            buf.copy_(proj.weight)

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def _lin(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        buf = self.weight_cache.get(id(proj))
        if buf is None or buf.dtype != x.dtype:
            return F.linear(x, proj.weight.to(x.dtype))
        return F.linear(x, _CachedCast.apply(proj.weight, buf))

    @torch.no_grad()
    def clean_pass(self, ids: torch.Tensor) -> tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
        base = self.base
        B, L = ids.shape
        x = base.model.embed_tokens(ids)
        cos, sin = base.model.rotary_emb(torch.arange(L, device=ids.device))
        attn_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(base.model.layers):
            h = layer.input_layernorm(x)
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                qkv = causal_conv1d(lin.in_proj_qkv(h), lin.conv1d.weight.squeeze(1))
                q, k, v = qkv.split([lin.key_dim, lin.key_dim, lin.value_dim], dim=-1)
                q = q.reshape(B, L, lin.num_k_heads, lin.head_k_dim)
                k = k.reshape(B, L, lin.num_k_heads, lin.head_k_dim)
                v = v.reshape(B, L, lin.num_v_heads, lin.head_v_dim)
                if lin.num_v_heads != lin.num_k_heads:
                    rep = lin.num_v_heads // lin.num_k_heads
                    q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
                beta = lin.in_proj_b(h).sigmoid()
                g = -lin.A_log.float().exp() * F.softplus(lin.in_proj_a(h).float() + lin.dt_bias)
                o, _ = gated_delta_rule_chunk(q, k, v, g, beta)
                attn_kv[i] = (apply_rotary(k, cos, sin), v)
                z = lin.in_proj_z(h).view(B, L, lin.num_v_heads, lin.head_v_dim)
                o = lin.norm(o.reshape(-1, lin.head_v_dim), z.reshape(-1, lin.head_v_dim)).view(B, L, lin.value_dim)
                x = x + lin.out_proj(o)
            else:
                att = layer.self_attn
                H, Hkv, D = att.num_heads, att.num_kv_heads, att.head_dim
                q, gate = att.q_proj(h).view(B, L, H, 2 * D).chunk(2, dim=-1)
                q = apply_rotary(att.q_norm(q), cos, sin)
                k = apply_rotary(att.k_norm(att.k_proj(h).view(B, L, Hkv, D)), cos, sin)
                v = att.v_proj(h).view(B, L, Hkv, D)
                attn_kv[i] = (k, v)
                o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                                   scale=att.scaling, enable_gqa=True).transpose(1, 2)
                o = o.reshape(B, L, H * D) * torch.sigmoid(gate.reshape(B, L, H * D))
                x = x + att.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return base.model.norm(x), attn_kv

    def mask_layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor,
                   clean_k: torch.Tensor, clean_v: torch.Tensor) -> torch.Tensor:
        layer = self.base.model.layers[i]
        B, N, _ = x.shape
        key = str(i)
        h = layer.input_layernorm(x)
        if layer.layer_type == LINEAR:
            lin = layer.linear_attn
            Hv, Dv = lin.num_v_heads, lin.head_v_dim
            q = apply_rotary(self._lin(self.delta_q[key], h).view(B, N, Hv, Dv), cos, sin)
            k = apply_rotary(self._lin(self.delta_k[key], h).view(B, N, Hv, Dv), cos, sin)
            v = self._lin(self.delta_v[key], h).view(B, N, Hv, Dv)
            keys = torch.cat((clean_k, k), dim=1)
            vals = torch.cat((clean_v, v), dim=1)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2), attn_mask=attn_mask,
                                               scale=Dv ** -0.5).transpose(1, 2)
            z = lin.in_proj_z(h).view(B, N, Hv, Dv)
            o = lin.norm(o.reshape(-1, Dv), z.reshape(-1, Dv)).view(B, N, lin.value_dim)
            x = x + lin.out_proj(o)
        else:
            att = layer.self_attn
            H, Hkv, D = att.num_heads, att.num_kv_heads, att.head_dim
            q, gate = self._lin(self.attn_q[key], h).view(B, N, H, 2 * D).chunk(2, dim=-1)
            q = apply_rotary(att.q_norm(q), cos, sin)
            k = apply_rotary(att.k_norm(self._lin(self.attn_k[key], h).view(B, N, Hkv, D)), cos, sin)
            v = self._lin(self.attn_v[key], h).view(B, N, Hkv, D)
            keys = torch.cat((clean_k, k), dim=1)
            vals = torch.cat((clean_v, v), dim=1)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2), attn_mask=attn_mask,
                                               scale=att.scaling, enable_gqa=True).transpose(1, 2)
            o = o.reshape(B, N, H * D) * torch.sigmoid(gate.reshape(B, N, H * D))
            x = x + att.o_proj(o)
        return x + layer.mlp(layer.post_attention_layernorm(x))

    def mask_pass(self, attn_kv: dict, mask_pos: torch.Tensor, attn_mask: torch.Tensor, B: int) -> torch.Tensor:
        x = self.mask_embed.to(self.dtype).view(1, 1, -1).expand(B, mask_pos.numel(), -1)
        cos, sin = self.base.model.rotary_emb(mask_pos)
        layers = self.compiled_mask_layers
        for i in range(len(self.base.model.layers)):
            ck, cv = attn_kv[i]
            fn = layers[i] if layers is not None else functools.partial(self.mask_layer, i)
            if self.grad_checkpoint and torch.is_grad_enabled():
                x = checkpoint(fn, x, cos, sin, attn_mask, ck, cv, use_reentrant=False)
            else:
                x = fn(x, cos, sin, attn_mask, ck, cv)
        return self.base.model.norm(x)

    def compile_layers(self) -> None:
        self.compiled_mask_layers = [torch.compile(functools.partial(self.mask_layer, i), dynamic=False)
                                     for i in range(len(self.base.model.layers))]

    def kl_loss(self, student_h: torch.Tensor, teacher_h: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.base.lm_head.weight
        B, N, _ = student_h.shape
        s = student_h.reshape(B * N, -1)
        t = teacher_h.reshape(B * N, -1)
        total = torch.zeros((), device=s.device)
        agree = torch.zeros((), device=s.device)
        for start in range(0, B * N, chunk):
            with torch.no_grad():
                tz = F.linear(t[start:start + chunk], w)
                tp = F.softmax(tz.float(), dim=-1).to(tz.dtype)
                ent = F.cross_entropy(tz, tp, reduction="sum")
            sz = F.linear(s[start:start + chunk], w)
            ce = F.cross_entropy(sz, tp, reduction="sum")
            total = total + (ce - ent)
            agree = agree + (tz.argmax(-1) == sz.detach().argmax(-1)).float().sum()
        return total / (B * N), agree / (B * N)


def build_anchor_layout(L: int, block: int, n_blocks: int, gen: torch.Generator, device) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks = min(n_blocks, L - block + 1)
    anchors = torch.randperm(L - block + 1, generator=gen)[:n_blocks].sort().values
    mask_pos = (anchors[:, None] + torch.arange(1, block, dtype=torch.long)[None, :]).reshape(-1)
    block_id = torch.arange(n_blocks, dtype=torch.long).repeat_interleave(block - 1)
    allow_clean = torch.arange(L, dtype=torch.long)[None, :] <= anchors[block_id][:, None]
    allow_block = block_id[:, None] == block_id[None, :]
    attn_mask = torch.cat((allow_clean, allow_block), dim=1)[None, None]
    return mask_pos.to(device), attn_mask.to(device)
