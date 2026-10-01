from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import sdpa_kernel

from export import load_export
from inference.types import PRECISIONS, Options, Result
from model.config import LINEAR
from model import metal
from model.model import (EFFICIENT, FrozenRMSNorm, RMSNorm, apply_rotary, conv_step, decode_attention, delta_gates,
                         gated_delta_rule_advance_inplace, gated_delta_rule_chunk, gated_delta_rule_step, metal_enabled, rotate_into_cache)
from drafter.view import DrafterView
from prep.format import DataFormat, Question

# SDPA's efficient kernel pads a copy of any mask whose row stride is not aligned, in every call.
MASK_ROW_ALIGNMENT = 16
# Longer passes run eagerly at their exact length: there the GPU work outweighs the launch cost a graph removes, and padding would add to it.
EXTEND_GRAPH_LENGTHS = (32, 64, 128, 256, 512, 1024, 1536, 2048)
MANY_QUESTIONS = 3


@dataclass(frozen=True)
class DecodeBlock:
    K: int
    J: int
    M: int
    ar_k: torch.Tensor
    ar_j: torch.Tensor
    mask_offset: torch.Tensor
    limit_offset: torch.Tensor
    allow_mm: torch.Tensor


def decode_block(K: int, device: torch.device) -> DecodeBlock:
    J = K - 1
    ar_k, ar_j = torch.arange(K, device=device), torch.arange(J, device=device)
    block_of_mask = ar_k.repeat_interleave(J)
    row_block = torch.cat((torch.full((K,), -1, device=device), block_of_mask))
    allow_mm = (row_block[:, None] == block_of_mask[None, :]) & (row_block[:, None] >= 0)
    return DecodeBlock(K, J, K * J, ar_k, ar_j, block_of_mask + (ar_j + 1).repeat(K), torch.cat((ar_k, block_of_mask)), allow_mm)


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


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    padded = -(-allowed.shape[-1] // MASK_ROW_ALIGNMENT) * MASK_ROW_ALIGNMENT
    mask = torch.full((*allowed.shape[:-1], padded), float("-inf"), dtype=dtype, device=allowed.device)[..., :allowed.shape[-1]]
    return mask.masked_fill_(allowed, 0.0)


class GroupRow(NamedTuple):
    record: DataFormat
    question: Question
    options: Options


def pack_requests(sizes: list[int], capacity: int) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    for request, size in enumerate(sizes):
        questions = [(request, i) for i in range(size)]
        for start in range(0, size, capacity):
            part = questions[start:start + capacity]
            group = next((g for g in groups if len(g) + len(part) <= capacity), None)
            if group is None:
                groups.append(part)
            else:
                group.extend(part)
    return groups


def common_prefix(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs) - 1
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs):
            return i
    return n


class Engine:
    def __init__(self, model: str, drafter: str, block: int = 4, precision: str = "bf16", max_rows: int = 8, max_len: int = 8192,
                 window_step: int = 512, sync_every: int = 1, device: str | None = None):
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
        if fp8:
            if self.mps:
                from inference.fp8_metal import load_view
            else:
                from inference.fp8 import load_view
            view, head, encoder = load_view(model, drafter, block, dev)
            base = view.base
            self.dtype = base.model.embed_tokens.weight.dtype
        else:
            base, head, encoder = load_export(model, device=dev)
            view = DrafterView(base, block=block).to(dev)
            loaded = view.load_state_dict(load_file(drafter, device=str(dev)), strict=False)
            missing = [k for k in loaded.missing_keys if not k.startswith("base.")]
            if loaded.unexpected_keys or missing:
                raise ValueError(f"drafter keys do not match: unexpected {loaded.unexpected_keys[:3]}, missing {missing[:3]}")
            view.requires_grad_(False)
            self.dtype = base.lm_head.weight.dtype
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
        # On CUDA, projections that read the same input run as one GEMM, which saves launches and reads the input once. In fp8, in_proj_b and in_proj_a stay bf16, so a delta layer's group falls back to merging just that pair.
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
                warm(nn.ModuleList(self.merged_projections.values()))
        self.log_decay_rates = {i: -layer.linear_attn.A_log.float().exp() for i, layer in enumerate(base.model.layers) if layer.layer_type == LINEAR}
        torch.accelerator.empty_cache()
        self.base, self.head, self.encoder, self.view = base, head, encoder, view
        self.cfg = cfg = base.cfg
        self.K = K = block
        self.J = K - 1
        self.M = K * self.J
        self.R, self.L = max_rows, max_len
        self.trash = max_len + self.M
        slots = max_len + self.M + 1
        if sync_every < 1:
            raise ValueError(f"sync_every must be at least 1, got {sync_every}")
        # A host sync leaves the GPU idle for a short gap in every cycle, while syncing every n cycles runs up to n - 1 full cycles after the last row finishes, so syncing every cycle wins unless a decode runs more cycles than about twice a cycle's time divided by a sync gap.
        self.window_step, self.sync_every = window_step, sync_every
        self.eos = encoder.think_end_id
        self.pad = encoder.pad_id
        self.opt_end = encoder.opt_end_id
        self.empty_think = list(encoder.empty_think)
        self.bias = torch.zeros(cfg.vocab_size, device=dev, dtype=self.dtype)
        self.bias[torch.tensor(encoder.banned_ids, device=dev)] = -1e4
        self.mask_embed = view.mask_embed.detach().to(self.dtype)
        self.layers = base.model.layers
        self.k, self.v, self.dk, self.dv, self.conv, self.rec = {}, {}, {}, {}, {}, {}
        delta = [layer.linear_attn for layer in self.layers if layer.layer_type == LINEAR]
        conv_dim, conv_kernel = delta[0].conv_dim, delta[0].conv_kernel
        # Every delta layer's conv state is a view of one buffer, and so is the conv input of a decode cycle, so that one gather commits them all.
        self.conv_states = torch.zeros(len(delta), max_rows, conv_dim, conv_kernel, device=dev, dtype=self.dtype)
        self.conv_inputs: dict[int, torch.Tensor] = {}
        self.conv_slot: dict[int, int] = {}
        for i, layer in enumerate(self.layers):
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                self.dk[i] = torch.zeros(max_rows, slots, lin.num_v_heads, lin.head_v_dim, device=dev, dtype=self.dtype)
                self.dv[i] = torch.zeros_like(self.dk[i])
                self.conv_slot[i] = len(self.conv_slot)
                self.conv[i] = self.conv_states[self.conv_slot[i]]
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
        self.ar_l = torch.arange(max_len + 1, device=dev)
        self.ar_conv_kernel = torch.arange(cfg.linear_conv_kernel_dim, device=dev)
        self.block = decode_block(K, dev)
        # Tuned on an M4 Pro with torch 2.14: in FP8, a cycle of MANY_QUESTIONS or more is compute-bound, so block 2's 4 rows per question
        # beat block 4's 16 rows even though block 2 accepts fewer tokens. In bf16 they do not, because MPS's bf16 matmul is slow at 9 to 15 rows.
        self.many_questions_block = decode_block(2, dev) if self.mps and fp8 and K > 2 else self.block
        for block_K in {self.block.K, self.many_questions_block.K}:
            self.conv_inputs[block_K] = torch.empty(len(delta), max_rows, conv_dim, conv_kernel + block_K, device=dev, dtype=self.dtype)
        self.cycle_graphs: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.extend_graphs: dict[tuple[int, int, int, bool], torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle() if self.cuda else None
        if self.cuda:
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
        return gated_delta_rule_step(q, k, v, g, beta, initial_state=state, output_final_state=False)[0]

    def delta_commit(self, k, v, g, beta, state, accepted) -> None:
        if self.metal:
            metal.gated_delta_rule_advance_inplace(k, v, g, beta, state, accepted)
        else:
            gated_delta_rule_advance_inplace(k, v, g, beta, state, accepted)

    def attention_mask(self, allowed: torch.Tensor) -> torch.Tensor:
        return additive_mask(allowed, self.dtype) if self.cuda else allowed

    def cached_attention(self, q, k_cache, v_cache, mask, scale: float, used_end: torch.Tensor | None, tail_start: int) -> torch.Tensor:
        B, gqa = q.shape[0], q.shape[2] != k_cache.shape[2]
        if self.metal:
            # Tuned on an M4 Pro with torch 2.14: the chunked kernel wins from 2 rows for the delta layers' mask queries, and from 4 rows for attention.
            if B >= (4 if gqa else 2):
                return metal.chunked_cached_attention(q, k_cache, v_cache, mask[:, 0], scale, used_end, tail_start)
            return metal.cached_attention(q, k_cache, v_cache, mask[:, 0], scale, used_end, tail_start)
        return decode_attention(q, k_cache, v_cache, mask, scale)

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

    def split_candidate_rows(self, h: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        hc, hm = h[:, :K], h[:, K:]
        # With more than one row both halves are strided: CUDA then runs a batched GEMM that reads the weights once per row, and MPS on torch 2.14 multiplies a strided input by the untransposed candidate weights about 2x slower.
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
        mask = self.attention_mask((self.ar_l[:Lk][None, None, :] <= pos[:, :, None])[:, None])
        cos, sin = base.model.rotary_emb(pos)
        conv_columns = (lens[:, None] + self.ar_conv_kernel[None])[:, None, :]
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
                beta, g = delta_gates(b_in, a_in, lin.dt_bias, self.log_decay_rates[i], valid)
                o = self.delta_extend(q, k, v, g, beta, self.rec[i][:B], commit)
                if commit:
                    torch.gather(ext, 2, conv_columns.expand(B, lin.conv_dim, -1), out=self.conv[i][:B])
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

    def cycle(self, B: int, Lw: int, block: DecodeBlock) -> None:
        base, view = self.base, self.view
        K, J, M = block.K, block.J, block.M
        n, done, cand = self.n[:B], self.done[:B], self.cand[:B, :K]
        pos_c = n[:, None] + block.ar_k[None]
        pos = torch.cat((pos_c, n[:, None] + block.mask_offset[None]), dim=1)
        cos, sin = base.model.rotary_emb(pos)
        cos_c, sin_c, cos_m, sin_m = (t.contiguous() for t in (cos[:, :K], sin[:, :K], cos[:, K:], sin[:, K:]))
        used_end = n + K if self.metal else None
        rows = torch.arange(B, device=self.device)[:, None].expand(B, K)
        limit = n[:, None] + block.limit_offset[None]
        allow_cache = self.ar_l[:Lw][None, None, :] <= limit[:, :, None]
        mask = self.attention_mask(torch.cat((allow_cache, block.allow_mm[None].expand(B, -1, -1)), dim=2)[:, None])
        x = torch.cat((base.model.embed_tokens(cand), self.mask_embed.view(1, 1, -1).expand(B, M, -1)), dim=1)
        pending = {}
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            key = str(i)
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                Hv, Dv = lin.num_v_heads, lin.head_v_dim
                hc, hm = self.split_candidate_rows(h, K)
                qkv, b_in, a_in = self.project(hc, lin.in_proj_qkv, lin.in_proj_b, lin.in_proj_a)
                _, q, k, v = conv_step(self.conv[i][:B], qkv, lin.conv1d.weight, lin.key_dim, lin.value_dim, lin.head_k_dim, Dv,
                                       Hv // lin.num_k_heads, ext_out=self.conv_inputs[K][self.conv_slot[i], :B])
                beta, g = delta_gates(b_in, a_in, lin.dt_bias, self.log_decay_rates[i])
                o_c = self.delta_outputs(q, k, v, g, beta, self.rec[i][:B])
                pending[i] = (k, v, g, beta)
                qm, km, vm = (t.view(B, M, Hv, Dv) for t in self.project(hm, view.delta_q[key], view.delta_k[key], view.delta_v[key]))
                qm = rotate_into_cache(k, v, qm, km, vm, cos_c, sin_c, cos_m, sin_m, self.dk[i], self.dv[i], rows, pos_c, Lw)
                o_m = self.cached_attention(qm, self.dk[i], self.dv[i], mask[:, :, K:], Dv ** -0.5, used_end, Lw)
                o = torch.cat((o_c, o_m), dim=1)
                z = lin.in_proj_z(h).view(B, K + M, Hv, Dv)
                o = lin.norm(o.reshape(-1, Dv), z.reshape(-1, Dv)).view(B, K + M, lin.value_dim)
                x = x + lin.out_proj(o)
            else:
                att = layer.self_attn
                H, Hkv, D = att.num_heads, att.num_kv_heads, att.head_dim
                hc, hm = self.split_candidate_rows(h, K)
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
        first_eos = torch.where(cand == self.eos, block.ar_k[None], K).amin(1)
        j = torch.minimum(j, first_eos + 1)
        j = torch.minimum(j, self.cap[:B] - self.n_out[:B])
        j = torch.where(done, 0, j)
        jm1 = (j - 1).clamp(min=0)
        nxt = torch.cat((pred.gather(1, jm1[:, None]), am.gather(1, K + jm1[:, None] * J + block.ar_j[None])), dim=1)
        self.out[:B].scatter_(1, self.n_out[:B, None] + block.ar_k[None], cand)
        for i, (k, v, g, beta) in pending.items():
            self.delta_commit(k, v, g, beta, self.rec[i][:B], j)
        conv_columns = (j[:, None] + self.ar_conv_kernel[None])[None, :, None, :]
        torch.gather(self.conv_inputs[K][:, :B], 3, conv_columns.expand(*self.conv_states[:, :B].shape), out=self.conv_states[:, :B])
        cand.copy_(torch.where(done[:, None], cand, nxt))
        done.logical_or_((first_eos < j) | (self.n_out[:B] + j >= self.cap[:B]))
        n.add_(j)
        self.n_out[:B].add_(j)

    def cycle_graph(self, B: int, Lw: int, block: DecodeBlock) -> torch.cuda.CUDAGraph:
        graph = self.cycle_graphs.get((B, Lw, block.K))
        if graph is not None:
            return graph
        saved = self.done[:B].clone()
        self.done[:B].fill_(True)
        graph = self.captured(lambda: self.cycle(B, Lw, block), warmups=2)
        self.done[:B].copy_(saved)
        self.cycle_graphs[(B, Lw, block.K)] = graph
        return graph

    def cycle_runner(self, B: int, Lw: int, block: DecodeBlock) -> Callable[[], None]:
        if self.cuda:
            return self.cycle_graph(B, Lw, block).replay
        return lambda: self.cycle(B, Lw, block)

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
        block = self.many_questions_block if B >= MANY_QUESTIONS else self.block
        while True:
            need = top + (self.sync_every + 1) * block.K + 1
            run_cycle = self.cycle_runner(Bp, min(self.L, -(-need // self.window_step) * self.window_step), block)
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
        return self.answer_batch([(record, opts)])[0]

    @torch.no_grad()
    def answer_batch(self, jobs: list[tuple[DataFormat, Options]]) -> list[list[Result]]:
        results: list[list[Result | None]] = [[None] * len(record.questions) for record, _ in jobs]
        # Thinking and no-think requests never share a group, so no-think rows hold no decode rows, and whole requests are packed together
        # where they fit, so that a request's questions keep sharing their prompt prefix.
        for thinking in (False, True):
            requests = [job for job, (_, opts) in enumerate(jobs) if opts.think == thinking]
            for group in pack_requests([len(jobs[job][0].questions) for job in requests], self.R):
                rows = [GroupRow(jobs[requests[r]][0], jobs[requests[r]][0].questions[i], jobs[requests[r]][1]) for r, i in group]
                for (r, i), result in zip(group, self.group(rows)):
                    results[requests[r]][i] = result
        return results

    def group(self, rows: list[GroupRow]) -> list[Result]:
        enc = self.encoder
        prompts = [enc.encode(enc.prompt_text(row.record, row.question)) for row in rows]
        rems = [enc.encode(enc.remainder_text(row.question)) for row in rows]
        B = len(rows)
        slack = self.K + self.J + 2
        for p, r in zip(prompts, rems):
            if len(p) + len(r) + len(self.empty_think) + slack > self.L:
                raise ValueError(f"input of {len(p) + len(r)} tokens exceeds the {self.L}-token limit")
        budgets = [row.options.max_think if row.options.think else 0 for row in rows]
        thresholds = [row.options.nothink_threshold for row in rows]
        caps = [max(0, min(budget, self.L - len(p) - len(r) - slack)) for budget, p, r in zip(budgets, prompts, rems)]
        wants_nothink = [t is not None or not c for t, c in zip(thresholds, caps)]
        P = common_prefix(prompts) if B > 1 else 0
        self.reset(self.decode_rows(B))
        if P > 0:
            self.extend([prompts[0][:P]], [0], commit=True)
            self.broadcast(B, P)
        tails = [self.empty_think + r for r in rems]
        first = nothink = None
        if not any(caps):
            hn = self.extend([p[P:] + t for p, t in zip(prompts, tails)], [P] * B, commit=False)
            nothink = [self.readout(hn[b, len(p) - P:], t) for b, (p, t) in enumerate(zip(prompts, tails))]
        else:
            h = self.extend([p[P:] for p in prompts], [P] * B, commit=True)
            last = h[torch.arange(B, device=self.device), torch.tensor([len(p) - P - 1 for p in prompts], device=self.device)]
            first = (self.vocab_logits(last) + self.bias).argmax(-1)
            if any(t is not None for t in thresholds) or not all(caps):
                hn = self.extend(tails, [len(p) for p in prompts], commit=False)
                nothink = [self.readout(hn[b], tails[b]) for b in range(B)]
                caps = [c if t is None or max(nothink[b]) < t else 0 for b, (c, t) in enumerate(zip(caps, thresholds))]
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
                       nothink_probs=nothink[b] if nothink is not None and wants_nothink[b] else None,
                       prompt_tokens=len(prompts[b]) - (P if b else 0))
                for b in range(B)]
