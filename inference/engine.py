from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from export import load_export
from inference.types import PRECISIONS, Options, Result
from model.config import LINEAR
from model import metal
from model.model import FrozenRMSNorm, RMSNorm, apply_rotary, gated_delta_rule_chunk, gated_delta_rule_step, metal_enabled
from drafter.view import DrafterView
from prep.format import DataFormat, Question

EFFICIENT = [SDPBackend.EFFICIENT_ATTENTION]
# Longer passes run eagerly at their exact length: there the GPU work outweighs the launch cost a graph removes, and padding would add to it.
EXTEND_GRAPH_LENGTHS = (32, 64, 128, 256, 512, 1024, 1536, 2048)


def delta_step(q, k, v, g, beta, state, final: bool):
    return gated_delta_rule_step(q, k, v, g, beta, initial_state=state, output_final_state=final)


def store_transposed(linear: nn.Linear) -> None:
    linear.weight = nn.Parameter(linear.weight.t().contiguous().t(), requires_grad=False)


def merge_linears_in_place(linears: tuple[nn.Module, ...]) -> nn.Module | None:
    if any(linear.bias is not None for linear in linears):
        return None
    if all(type(linear) is nn.Linear for linear in linears):
        sizes = [linear.out_features for linear in linears]
        weight = torch.cat([linear.weight.detach() for linear in linears])
        merged = nn.Linear(linears[0].in_features, sum(sizes), bias=False, device="meta")
        merged.weight = nn.Parameter(weight, requires_grad=False)
        for linear, part in zip(linears, weight.split(sizes)):
            linear.weight = nn.Parameter(part, requires_grad=False)
        return merged
    from inference.fp8 import FP8Linear
    return FP8Linear.concatenated(list(linears)) if all(type(linear) is FP8Linear for linear in linears) else None


def freeze_rms_norms(module: nn.Module) -> None:
    for name, child in module.named_children():
        if type(child) is RMSNorm:
            setattr(module, name, FrozenRMSNorm(child))
        else:
            freeze_rms_norms(child)


def common_prefix(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs) - 1
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs):
            return i
    return n


class Engine:
    def __init__(self, model: str, drafter: str, block: int = 4, precision: str = "bf16", max_rows: int = 8, max_len: int = 8192,
                 window_step: int = 512, sync_every: int | None = None, device: str | None = None):
        self.device = dev = torch.device(device) if device else torch.accelerator.current_accelerator()
        self.mps, self.cuda = dev.type == "mps", dev.type == "cuda"
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        self.precision = precision
        fp8 = precision == "fp8"
        fp8_supported = self.metal if self.mps else (self.cuda and torch.cuda.get_device_capability(dev) >= (8, 9))
        if fp8 and not fp8_supported:
            raise ValueError("precision='fp8' needs a CUDA GPU with compute capability 8.9 or higher, or MPS with the Metal kernels enabled "
                             "(unset QWEN35_KERNELS=0)")
        base, head, encoder = load_export(model, device=dev)
        view = DrafterView(base, block=block).to(dev)
        loaded = view.load_state_dict(load_file(drafter, device=str(dev)), strict=False)
        missing = [k for k in loaded.missing_keys if not k.startswith("base.")]
        if loaded.unexpected_keys or missing:
            raise ValueError(f"drafter keys do not match: unexpected {loaded.unexpected_keys[:3]}, missing {missing[:3]}")
        view.requires_grad_(False)
        self.dtype = base.lm_head.weight.dtype
        if fp8 and self.mps:
            from inference.fp8_metal import quantize
            quantize(view)
        elif fp8:
            from inference.fp8 import quantize
            quantize(view)
        else:
            for proj in view.projections():
                proj.to(self.dtype)
        if self.mps and not fp8 and torch.backends.mps.is_macos_or_newer(15, 0):
            # On torch 2.14, MPS computes x @ W.T faster, and bitwise equal, with W.T contiguous once x has 10 or more rows, but slower with fewer; so only projections fed the K + M cycle rows switch, except attn_k/attn_v, which are slower even at M rows.
            for layer in base.model.layers:
                mixer = (layer.linear_attn.in_proj_z, layer.linear_attn.out_proj) if layer.layer_type == LINEAR else (layer.self_attn.o_proj,)
                for proj in (*mixer, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                    store_transposed(proj)
            for proj in (*view.delta_q.values(), *view.delta_k.values(), *view.delta_v.values(), *view.attn_q.values()):
                store_transposed(proj)
        # On CUDA, projections that read the same input run as one GEMM, which saves launches and reads the input once. In fp8, in_proj_b
        # and in_proj_a stay bf16, so a delta layer's group falls back to merging just that pair.
        self.merged_projections: dict[tuple[int, ...], nn.Module] = {}
        if self.cuda:
            groups = [(layer.linear_attn.in_proj_qkv, layer.linear_attn.in_proj_b, layer.linear_attn.in_proj_a) if layer.layer_type == LINEAR
                      else (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj) for layer in base.model.layers]
            groups += [(view.delta_q[key], view.delta_k[key], view.delta_v[key]) for key in view.delta_q]
            groups += [(view.attn_q[key], view.attn_k[key], view.attn_v[key]) for key in view.attn_q]
            for group in groups:
                for candidate in (group, group[1:]):
                    if (merged := merge_linears_in_place(candidate)) is not None:
                        self.merged_projections[tuple(map(id, candidate))] = merged
                        break
            freeze_rms_norms(base)
            if fp8:
                from inference.fp8 import warm
                warm(view)
                warm(nn.ModuleList(self.merged_projections.values()))
        self.log_decay_rates = {i: -layer.linear_attn.A_log.float().exp() for i, layer in enumerate(base.model.layers) if layer.layer_type == LINEAR}
        torch.accelerator.empty_cache()
        self.base, self.head, self.encoder, self.view = base, head, encoder, view
        self.cfg = cfg = base.cfg
        self.K = K = block
        self.J = J = K - 1
        self.M = K * J
        self.R, self.L = max_rows, max_len
        self.trash = max_len + self.M
        slots = max_len + self.M + 1
        # Without CUDA graphs a host sync is cheap, and each unneeded cycle after the last row finishes costs a full forward.
        self.window_step, self.sync_every = window_step, sync_every or (4 if self.cuda else 1)
        self.eos = encoder.think_end_id
        self.pad = encoder.pad_id
        self.opt_end = encoder.opt_end_id
        self.empty_think = list(encoder.empty_think)
        self.bias = torch.zeros(cfg.vocab_size, device=dev, dtype=self.dtype)
        self.bias[torch.tensor(encoder.banned_ids, device=dev)] = -1e4
        self.mask_embed = view.mask_embed.detach().to(self.dtype)
        self.layers = base.model.layers
        self.k, self.v, self.dk, self.dv, self.conv, self.rec = {}, {}, {}, {}, {}, {}
        for i, layer in enumerate(self.layers):
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                self.dk[i] = torch.zeros(max_rows, slots, lin.num_v_heads, lin.head_v_dim, device=dev, dtype=self.dtype)
                self.dv[i] = torch.zeros_like(self.dk[i])
                self.conv[i] = torch.zeros(max_rows, lin.conv_dim, lin.conv_kernel, device=dev, dtype=self.dtype)
                self.rec[i] = torch.zeros(max_rows, lin.num_v_heads, lin.head_k_dim, lin.head_v_dim, device=dev, dtype=torch.float32)
            else:
                att = layer.self_attn
                self.k[i] = torch.zeros(max_rows, slots, att.num_kv_heads, att.head_dim, device=dev, dtype=self.dtype)
                self.v[i] = torch.zeros_like(self.k[i])
        self.n = torch.zeros(max_rows, dtype=torch.long, device=dev)
        self.n_out = torch.zeros(max_rows, dtype=torch.long, device=dev)
        self.cap = torch.zeros(max_rows, dtype=torch.long, device=dev)
        self.done = torch.ones(max_rows, dtype=torch.bool, device=dev)
        self.cand = torch.zeros(max_rows, K, dtype=torch.long, device=dev)
        self.out = torch.zeros(max_rows, max_len + K, dtype=torch.long, device=dev)
        self.ar_k = torch.arange(K, device=dev)
        self.ar_j = torch.arange(J, device=dev)
        self.ar_l = torch.arange(max_len + 1, device=dev)
        block_of_mask = torch.arange(K, device=dev).repeat_interleave(J)
        self.mask_offset = block_of_mask + (self.ar_j + 1).repeat(K)
        self.limit_offset = torch.cat((self.ar_k, block_of_mask))
        row_block = torch.cat((torch.full((K,), -1, device=dev), block_of_mask))
        self.allow_mm = (row_block[:, None] == block_of_mask[None, :]) & (row_block[:, None] >= 0)
        self.cycle_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.extend_graphs: dict[tuple[int, int, int, bool], torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle() if self.cuda else None
        if self.cuda:
            from inference.fp8 import LARGE_M
            # CUDA's fp8 GEMM quantizes activations above LARGE_M rows, so a padded pass must not cross that limit where the real one would not.
            self.fp8_activation_rows = LARGE_M if fp8 else None
            self.capture_stream = torch.cuda.Stream()
            longest = min(EXTEND_GRAPH_LENGTHS[-1], max_len)
            self.extend_ids = torch.full((max_rows, longest), self.pad, dtype=torch.long, device=dev)
            self.extend_valid = torch.zeros(max_rows, longest, dtype=torch.bool, device=dev)
            self.extend_lens = torch.zeros(max_rows, dtype=torch.long, device=dev)
            self.extend_starts = torch.zeros(max_rows, dtype=torch.long, device=dev)
            self.extend_out = torch.empty(max_rows, longest, cfg.hidden_size, dtype=self.dtype, device=dev)

    @property
    def metal(self) -> bool:
        return self.mps and metal_enabled()

    def delta_extend(self, q, k, v, g, beta, state, commit: bool) -> torch.Tensor:
        if self.metal:
            return metal.gated_delta_rule_outputs(q, k, v, g, beta, state, advance_state=commit)
        o, final = gated_delta_rule_chunk(q, k, v, g, beta, initial_state=state, output_final_state=True)
        if commit:
            state.copy_(final)
        return o

    def delta_outputs(self, q, k, v, g, beta, state) -> torch.Tensor:
        if self.metal:
            return metal.gated_delta_rule_outputs(q, k, v, g, beta, state)
        return delta_step(q, k, v, g, beta, state, False)[0]

    def delta_commit(self, k, v, g, beta, state, accepted, keep) -> None:
        if self.metal:
            metal.gated_delta_rule_advance_inplace(k, v, g, beta, state, accepted)
        else:
            state.copy_(delta_step(torch.zeros_like(k), k, v, g * keep, beta * keep.to(beta.dtype), state, True)[1])

    def cached_attention(self, q, k_cache, v_cache, mask, scale: float, used_end: torch.Tensor | None, tail_start: int) -> torch.Tensor:
        B, length = q.shape[0], mask.shape[-1]
        gqa = q.shape[2] != k_cache.shape[2]
        if self.metal:
            # Tuned on an M4 Pro with torch 2.14: the chunked kernel wins from 2 rows for the delta layers' mask queries, and from 4 rows for attention.
            if B >= (4 if gqa else 2):
                return metal.chunked_cached_attention(q, k_cache, v_cache, mask[:, 0], scale, used_end, tail_start)
            return metal.cached_attention(q, k_cache, v_cache, mask[:, 0], scale, used_end, tail_start)
        keys, vals = k_cache[:B, :length], v_cache[:B, :length]
        with sdpa_kernel(EFFICIENT):
            return F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2), attn_mask=mask, scale=scale,
                                                  enable_gqa=gqa).transpose(1, 2)

    def project(self, x: torch.Tensor, *linears: nn.Module) -> tuple[torch.Tensor, ...]:
        if not self.merged_projections:
            return tuple(linear(x) for linear in linears)
        outputs, start = [], 0
        while start < len(linears):
            end = next((end for end in range(len(linears), start + 1, -1) if tuple(map(id, linears[start:end])) in self.merged_projections), start + 1)
            if end == start + 1:
                outputs.append(linears[start](x))
            else:
                merged = self.merged_projections[tuple(map(id, linears[start:end]))]
                outputs += merged(x).split([linear.out_features for linear in linears[start:end]], dim=-1)
            start = end
        return tuple(outputs)

    def split_candidate_rows(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hc, hm = h[:, :self.K], h[:, self.K:]
        # With more than one row both halves are strided. CUDA multiplies a strided input with a batched GEMM that reads the weights once per
        # row, and MPS multiplies a strided input by the untransposed candidate weights about 2x slower.
        if self.cuda:
            return hc.contiguous(), hm.contiguous()
        return (hc.contiguous() if self.mps else hc), hm

    def vocab_logits(self, h: torch.Tensor) -> torch.Tensor:
        if not self.mps or not isinstance(self.base.lm_head, nn.Linear):
            return self.base.lm_head(h)
        # MPS in torch 2.14 picks a slow kernel for the 248k-row matmul; row chunks of the weight give bitwise-equal logits much faster.
        return torch.cat([F.linear(h, w) for w in self.base.lm_head.weight.chunk(8)], dim=-1)

    def decode_rows(self, n: int) -> int:
        if not self.cuda:
            return n
        # CUDA captures a graph per row count, so rounding up to a power of two keeps the number of graphs small.
        b = 1
        while b < n:
            b *= 2
        return min(b, self.R)

    @torch.no_grad()
    def extend(self, seqs: list[list[int]], starts: list[int], commit: bool) -> torch.Tensor:
        B, T = len(seqs), max(len(s) for s in seqs)
        Lk = max(st + len(s) for st, s in zip(starts, seqs))
        padded_T = next((n for n in EXTEND_GRAPH_LENGTHS if T <= n <= self.L), None) if self.cuda else None
        # Short passes round their row count up like decode cycles, so fewer graphs are captured; on long passes the extra rows would cost real GPU time.
        rows = self.decode_rows(B) if padded_T is not None and padded_T <= 256 else B
        limit = self.fp8_activation_rows if self.cuda else None
        if padded_T is not None and limit is not None and (B * T > limit) != (rows * padded_T > limit):
            padded_T = None
        width = T if padded_T is None else padded_T
        ids = torch.full((B, width), self.pad, dtype=torch.long)
        valid = torch.zeros(B, width, dtype=torch.bool)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            valid[b, :len(s)] = True
        lens, start_positions = torch.tensor([len(s) for s in seqs]), torch.tensor(starts)
        if padded_T is None:
            dev = self.device
            return self.extend_pass(ids.to(dev), valid.to(dev), lens.to(dev), start_positions.to(dev), Lk, commit)
        padded_Lk = padded_T if max(starts) == 0 else min(max(self.window_step, 1 << (Lk - 1).bit_length()), self.L)
        self.extend_ids[:rows, :padded_T].fill_(self.pad)
        self.extend_valid[:rows, :padded_T].zero_()
        self.extend_lens[:rows].zero_()
        self.extend_starts[:rows].zero_()
        self.extend_ids[:B, :padded_T].copy_(ids)
        self.extend_valid[:B, :padded_T].copy_(valid)
        self.extend_lens[:B].copy_(lens)
        self.extend_starts[:B].copy_(start_positions)
        self.extend_graph(rows, padded_T, padded_Lk, commit).replay()
        # A view of the shared output buffer: the next extend overwrites it.
        return self.extend_out[:B, :T]

    def extend_graph(self, B: int, T: int, Lk: int, commit: bool) -> torch.cuda.CUDAGraph:
        graph = self.extend_graphs.get((B, T, Lk, commit))
        if graph is not None:
            return graph
        inputs = (self.extend_ids[:B, :T], self.extend_valid[:B, :T], self.extend_lens[:B], self.extend_starts[:B], Lk, commit)
        # A committing warm-up advances conv and rec, so they are restored before the caller's replay.
        states = [t[:B] for d in (self.conv, self.rec) for t in d.values()] if commit else []
        saved = [t.clone() for t in states]
        graph = self.captured(lambda: self.extend_out[:B, :T].copy_(self.extend_pass(*inputs)), warmups=1)
        for t, s in zip(states, saved):
            t.copy_(s)
        self.extend_graphs[(B, T, Lk, commit)] = graph
        return graph

    def captured(self, run: Callable[[], object], warmups: int) -> torch.cuda.CUDAGraph:
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.capture_stream):
            for _ in range(warmups):
                run()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, stream=self.capture_stream):
            run()
        return graph

    def extend_pass(self, ids: torch.Tensor, valid: torch.Tensor, lens: torch.Tensor, starts: torch.Tensor, Lk: int, commit: bool) -> torch.Tensor:
        dev, base = self.device, self.base
        B, T = ids.shape
        pos = starts[:, None] + torch.arange(T, device=dev)[None]
        wpos = torch.where(valid, pos, self.trash)
        rows = torch.arange(B, device=dev)[:, None].expand(B, T)
        mask = (self.ar_l[:Lk][None, None, :] <= pos[:, :, None])[:, None]
        cos, sin = base.model.rotary_emb(pos)
        keep = valid.float()[..., None]
        x = base.model.embed_tokens(ids)
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                qkv, b_in, a_in = self.project(h, lin.in_proj_qkv, lin.in_proj_b, lin.in_proj_a)
                ext = torch.cat((self.conv[i][:B], qkv.transpose(1, 2)), dim=-1)
                conv = F.silu(F.conv1d(ext, lin.conv1d.weight, groups=lin.conv_dim)[..., -T:]).transpose(1, 2)
                q, k, v = conv.split([lin.key_dim, lin.key_dim, lin.value_dim], dim=-1)
                rep = lin.num_v_heads // lin.num_k_heads
                q = q.reshape(B, T, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                k = k.reshape(B, T, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                v = v.reshape(B, T, lin.num_v_heads, lin.head_v_dim)
                beta = b_in.sigmoid() * keep.to(h.dtype)
                g = self.log_decay_rates[i] * F.softplus(a_in.float() + lin.dt_bias) * keep
                o = self.delta_extend(q, k, v, g, beta, self.rec[i][:B], commit)
                if commit:
                    idx = (lens[:, None] + torch.arange(lin.conv_kernel, device=dev)[None])[:, None, :].expand(B, lin.conv_dim, -1)
                    self.conv[i][:B].copy_(ext.gather(2, idx))
                self.dk[i][rows, wpos] = apply_rotary(k, cos, sin)
                self.dv[i][rows, wpos] = v
                z = lin.in_proj_z(h).view(B, T, lin.num_v_heads, lin.head_v_dim)
                o = lin.norm(o.reshape(-1, lin.head_v_dim), z.reshape(-1, lin.head_v_dim)).view(B, T, lin.value_dim)
                x = x + lin.out_proj(o)
            else:
                att = layer.self_attn
                H, Hkv, D = att.num_heads, att.num_kv_heads, att.head_dim
                q_in, k_in, v_in = self.project(h, att.q_proj, att.k_proj, att.v_proj)
                q, gate = q_in.view(B, T, H, 2 * D).chunk(2, dim=-1)
                q = apply_rotary(att.q_norm(q), cos, sin)
                self.k[i][rows, wpos] = apply_rotary(att.k_norm(k_in.view(B, T, Hkv, D)), cos, sin)
                self.v[i][rows, wpos] = v_in.view(B, T, Hkv, D)
                keys, vals = self.k[i][:B, :Lk], self.v[i][:B, :Lk]
                with sdpa_kernel(EFFICIENT):
                    o = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2), attn_mask=mask,
                                                       scale=att.scaling, enable_gqa=True).transpose(1, 2)
                o = o.reshape(B, T, H * D) * torch.sigmoid(gate.reshape(B, T, H * D))
                x = x + att.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return base.model.norm(x)

    def cycle(self, B: int, Lw: int) -> None:
        base, view = self.base, self.view
        K, J, M = self.K, self.J, self.M
        n, done, cand = self.n[:B], self.done[:B], self.cand[:B]
        pos_c = n[:, None] + self.ar_k[None]
        pos = torch.cat((pos_c, n[:, None] + self.mask_offset[None]), dim=1)
        cos, sin = base.model.rotary_emb(pos)
        cos_c, sin_c, cos_m, sin_m = (t.contiguous() for t in (cos[:, :K], sin[:, :K], cos[:, K:], sin[:, K:]))
        used_end = n + K if self.metal else None
        rows = torch.arange(B, device=self.device)[:, None].expand(B, K)
        limit = n[:, None] + self.limit_offset[None]
        allow_cache = self.ar_l[:Lw][None, None, :] <= limit[:, :, None]
        mask = torch.cat((allow_cache, self.allow_mm[None].expand(B, -1, -1)), dim=2)[:, None]
        x = torch.cat((base.model.embed_tokens(cand), self.mask_embed.view(1, 1, -1).expand(B, M, -1)), dim=1)
        pending = {}
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            key = str(i)
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                Hv, Dv = lin.num_v_heads, lin.head_v_dim
                hc, hm = self.split_candidate_rows(h)
                qkv, b_in, a_in = self.project(hc, lin.in_proj_qkv, lin.in_proj_b, lin.in_proj_a)
                ext = torch.cat((self.conv[i][:B], qkv.transpose(1, 2)), dim=-1)
                conv = F.silu(F.conv1d(ext, lin.conv1d.weight, groups=lin.conv_dim)[..., -K:]).transpose(1, 2)
                q, k, v = conv.split([lin.key_dim, lin.key_dim, lin.value_dim], dim=-1)
                rep = lin.num_v_heads // lin.num_k_heads
                q = q.reshape(B, K, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                k = k.reshape(B, K, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                v = v.reshape(B, K, Hv, Dv).contiguous()
                beta = b_in.sigmoid()
                g = self.log_decay_rates[i] * F.softplus(a_in.float() + lin.dt_bias)
                o_c = self.delta_outputs(q, k, v, g, beta, self.rec[i][:B])
                pending[i] = (k, v, g, beta, ext)
                self.dk[i][rows, pos_c] = apply_rotary(k, cos_c, sin_c)
                self.dv[i][rows, pos_c] = v
                qm, km, vm = (t.view(B, M, Hv, Dv) for t in self.project(hm, view.delta_q[key], view.delta_k[key], view.delta_v[key]))
                qm, km = apply_rotary(qm, cos_m, sin_m), apply_rotary(km, cos_m, sin_m)
                self.dk[i][:B, Lw:Lw + M] = km
                self.dv[i][:B, Lw:Lw + M] = vm
                o_m = self.cached_attention(qm, self.dk[i], self.dv[i], mask[:, :, K:], Dv ** -0.5, used_end, Lw)
                o = torch.cat((o_c, o_m), dim=1)
                z = lin.in_proj_z(h).view(B, K + M, Hv, Dv)
                o = lin.norm(o.reshape(-1, Dv), z.reshape(-1, Dv)).view(B, K + M, lin.value_dim)
                x = x + lin.out_proj(o)
            else:
                att = layer.self_attn
                H, Hkv, D = att.num_heads, att.num_kv_heads, att.head_dim
                hc, hm = self.split_candidate_rows(h)
                qc_in, kc_in, vc_in = self.project(hc, att.q_proj, att.k_proj, att.v_proj)
                qm_in, km_in, vm_in = self.project(hm, view.attn_q[key], view.attn_k[key], view.attn_v[key])
                qc, gc = qc_in.view(B, K, H, 2 * D).chunk(2, dim=-1)
                qm, gm = qm_in.view(B, M, H, 2 * D).chunk(2, dim=-1)
                q = apply_rotary(att.q_norm(torch.cat((qc, qm), 1)), cos, sin)
                kc = att.k_norm(kc_in.view(B, K, Hkv, D))
                km = att.k_norm(km_in.view(B, M, Hkv, D))
                k = apply_rotary(torch.cat((kc, km), 1), cos, sin)
                self.k[i][rows, pos_c] = k[:, :K]
                self.v[i][rows, pos_c] = vc_in.view(B, K, Hkv, D)
                self.k[i][:B, Lw:Lw + M] = k[:, K:]
                self.v[i][:B, Lw:Lw + M] = vm_in.view(B, M, Hkv, D)
                o = self.cached_attention(q, self.k[i], self.v[i], mask, att.scaling, used_end, Lw)
                o = o.reshape(B, K + M, H * D) * torch.sigmoid(torch.cat((gc, gm), 1).reshape(B, K + M, H * D))
                x = x + att.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        am = (self.vocab_logits(base.model.norm(x)) + self.bias).argmax(-1)
        pred = am[:, :K]
        match = (cand[:, 1:] == pred[:, :-1]).long()
        j = torch.cumprod(match, 1).sum(1) + 1
        first_eos = torch.where(cand == self.eos, self.ar_k[None], K).amin(1)
        j = torch.minimum(j, first_eos + 1)
        j = torch.minimum(j, self.cap[:B] - self.n_out[:B])
        j = torch.where(done, 0, j)
        jm1 = (j - 1).clamp(min=0)
        nxt = torch.cat((pred.gather(1, jm1[:, None]), am.gather(1, K + jm1[:, None] * J + self.ar_j[None])), dim=1)
        self.out[:B].scatter_(1, self.n_out[:B, None] + self.ar_k[None], cand)
        keep = None if self.metal else (self.ar_k[None] < j[:, None]).float()[..., None]
        for i, (k, v, g, beta, ext) in pending.items():
            self.delta_commit(k, v, g, beta, self.rec[i][:B], j, keep)
            idx = (j[:, None] + torch.arange(ext.shape[-1] - K, device=self.device)[None])[:, None, :].expand(B, ext.shape[1], -1)
            self.conv[i][:B].copy_(ext.gather(2, idx))
        cand.copy_(torch.where(done[:, None], cand, nxt))
        done.logical_or_((first_eos < j) | (self.n_out[:B] + j >= self.cap[:B]))
        n.add_(j)
        self.n_out[:B].add_(j)

    def cycle_graph(self, B: int, Lw: int) -> torch.cuda.CUDAGraph:
        graph = self.cycle_graphs.get((B, Lw))
        if graph is not None:
            return graph
        saved = self.done[:B].clone()
        self.done[:B].fill_(True)
        graph = self.captured(lambda: self.cycle(B, Lw), warmups=2)
        self.done[:B].copy_(saved)
        self.cycle_graphs[(B, Lw)] = graph
        return graph

    def cycle_runner(self, B: int, Lw: int) -> Callable[[], None]:
        if self.cuda:
            return self.cycle_graph(B, Lw).replay
        return lambda: self.cycle(B, Lw)

    @torch.no_grad()
    def decode(self, B: int, prompts: list[list[int]], first: torch.Tensor, caps: list[int]) -> tuple[list[list[int]], list[bool]]:
        Bp = self.decode_rows(B)
        dev = self.device
        self.n[:Bp].zero_()
        self.n[:B].copy_(torch.tensor([len(p) for p in prompts], device=dev))
        self.n_out[:Bp].zero_()
        self.cap[:Bp].zero_()
        self.cap[:B].copy_(torch.tensor(caps, device=dev))
        self.done[:Bp].fill_(True)
        self.done[:B].copy_(self.cap[:B] == 0)
        self.cand[:Bp].zero_()
        self.cand[:B, 0].copy_(first)
        top = max(len(p) for p in prompts)
        while True:
            need = top + (self.sync_every + 1) * self.K + 1
            run_cycle = self.cycle_runner(Bp, min(self.L, -(-need // self.window_step) * self.window_step))
            for _ in range(self.sync_every):
                run_cycle()
            finished, top = torch.stack((self.done[:Bp].all().long(), self.n[:B].max())).tolist()
            if finished:
                break
        n_out = self.n_out[:B].tolist()
        out = self.out[:B].tolist()
        chains = [out[b][:n_out[b]] for b in range(B)]
        return chains, [bool(c) and c[-1] == self.eos for c in chains]

    def readout(self, h: torch.Tensor, seq: list[int]) -> list[float]:
        opts = [t for t, tok in enumerate(seq) if tok == self.opt_end]
        logits = self.head(h[opts][None], h[len(seq) - 1][None]).float()
        return F.softmax(logits[0], dim=-1).tolist()

    def reset(self, B: int) -> None:
        for i in self.conv:
            self.conv[i][:B].zero_()
            self.rec[i][:B].zero_()

    def broadcast(self, B: int, P: int) -> None:
        if B == 1:
            return
        for d in (self.k, self.v, self.dk, self.dv):
            for t in d.values():
                t[1:B, :P].copy_(t[:1, :P].expand(B - 1, -1, -1, -1))
        for d in (self.conv, self.rec):
            for t in d.values():
                t[1:B].copy_(t[:1].expand(B - 1, *t.shape[1:]))

    @torch.no_grad()
    def answer(self, record: DataFormat, opts: Options) -> list[Result]:
        out: list[Result] = []
        for s in range(0, len(record.questions), self.R):
            out.extend(self.group(record, record.questions[s:s + self.R], opts))
        return out

    def group(self, record: DataFormat, questions: list[Question], opts: Options) -> list[Result]:
        enc = self.encoder
        prompts = [enc.encode(enc.prompt_text(record, q)) for q in questions]
        rems = [enc.encode(enc.remainder_text(q)) for q in questions]
        B = len(questions)
        slack = self.K + self.J + 2
        for p, r in zip(prompts, rems):
            if len(p) + len(r) + len(self.empty_think) + slack > self.L:
                raise ValueError(f"input of {len(p) + len(r)} tokens exceeds the {self.L}-token limit")
        budget = opts.max_think if opts.think else 0
        caps = [max(0, min(budget, self.L - len(p) - len(r) - slack)) for p, r in zip(prompts, rems)]
        # CUDA's fp8 GEMM quantizes activations above LARGE_M rows, so one merged pass would move the last prompt token and the no-think tail
        # of a long prompt from bf16 to fp8 activations.
        merge_passes = not (self.cuda and self.precision == "fp8")
        P = common_prefix(prompts) if B > 1 or not merge_passes else 0
        self.reset(self.decode_rows(B))
        if P > 0:
            self.extend([prompts[0][:P]], [0], commit=True)
            self.broadcast(B, P)
        tails = [self.empty_think + r for r in rems]
        first = nothink = None
        if merge_passes and not any(caps):
            hn = self.extend([p[P:] + t for p, t in zip(prompts, tails)], [P] * B, commit=False)
            nothink = [self.readout(hn[b, len(p) - P:], t) for b, (p, t) in enumerate(zip(prompts, tails))]
        else:
            h = self.extend([p[P:] for p in prompts], [P] * B, commit=True)
            last = h[torch.arange(B, device=self.device), torch.tensor([len(p) - P - 1 for p in prompts], device=self.device)]
            first = (self.vocab_logits(last) + self.bias).argmax(-1)
            if opts.nothink_threshold is not None or not all(caps):
                hn = self.extend(tails, [len(p) for p in prompts], commit=False)
                nothink = [self.readout(hn[b], tails[b]) for b in range(B)]
                if opts.nothink_threshold is not None:
                    caps = [c if max(nothink[b]) < opts.nothink_threshold else 0 for b, c in enumerate(caps)]
        chains, closed = [[] for _ in range(B)], [False] * B
        probs = list(nothink) if nothink is not None else [[] for _ in range(B)]
        if any(caps):
            chains, closed = self.decode(B, prompts, first, caps)
            seqs = [r[1:] if c else r for r, c in zip(rems, closed)]
            hr = self.extend(seqs, [len(p) + len(ch) for p, ch in zip(prompts, chains)], commit=False)
            for b in range(B):
                if caps[b]:
                    probs[b] = self.readout(hr[b], seqs[b])
        return [Result(probs=probs[b], thought=bool(caps[b]), chain=chains[b], closed=closed[b],
                       nothink_probs=nothink[b] if nothink is not None else None, prompt_tokens=len(prompts[b]) - (P if b else 0))
                for b in range(B)]
