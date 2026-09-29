from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from export import load_export
from inference.types import Options, Result
from model.config import LINEAR
from inference import metal
from model.model import apply_rotary, gated_delta_rule_chunk, gated_delta_rule_step
from drafter.view import DrafterView
from prep.format import DataFormat, Question

EFFICIENT = [SDPBackend.EFFICIENT_ATTENTION]


def delta_step(q, k, v, g, beta, state, final: bool):
    return gated_delta_rule_step(q, k, v, g, beta, initial_state=state, output_final_state=final)


def common_prefix(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs) - 1
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs):
            return i
    return n


class Engine:
    def __init__(self, model: str, drafter: str, block: int = 4, fp8: bool = True, max_rows: int = 8, max_len: int = 8192,
                 window_step: int = 512, sync_every: int | None = None, device: str | None = None):
        self.device = dev = torch.device(device) if device else torch.accelerator.current_accelerator()
        base, head, encoder = load_export(model, device=dev)
        view = DrafterView(base, block=block).to(dev)
        loaded = view.load_state_dict(load_file(drafter, device=str(dev)), strict=False)
        missing = [k for k in loaded.missing_keys if not k.startswith("base.")]
        if loaded.unexpected_keys or missing:
            raise ValueError(f"drafter keys do not match: unexpected {loaded.unexpected_keys[:3]}, missing {missing[:3]}")
        view.requires_grad_(False)
        self.dtype = base.lm_head.weight.dtype
        fp8 = fp8 and dev.type == "cuda" and torch.cuda.get_device_capability(dev) >= (8, 9)
        if fp8:
            from inference.fp8 import quantize, warm
            quantize(view)
            warm(view)
        else:
            for proj in view.projections():
                proj.to(self.dtype)
        torch.accelerator.empty_cache()
        self.base, self.head, self.encoder, self.view = base, head, encoder, view
        self.fp8 = fp8
        self.cfg = cfg = base.cfg
        self.K = K = block
        self.J = J = K - 1
        self.M = K * J
        self.R, self.L = max_rows, max_len
        self.trash = max_len + self.M
        slots = max_len + self.M + 1
        # Without CUDA graphs a host sync is cheap, and each unneeded cycle after the last row finishes costs a full forward.
        self.window_step, self.sync_every = window_step, sync_every or (4 if dev.type == "cuda" else 1)
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
        self.all_steps = torch.full((max_rows,), K, dtype=torch.long, device=dev)
        block_of_mask = torch.arange(K, device=dev).repeat_interleave(J)
        self.mask_offset = block_of_mask + (self.ar_j + 1).repeat(K)
        self.limit_offset = torch.cat((self.ar_k, block_of_mask))
        row_block = torch.cat((torch.full((K,), -1, device=dev), block_of_mask))
        self.allow_mm = (row_block[:, None] == block_of_mask[None, :]) & (row_block[:, None] >= 0)
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle() if dev.type == "cuda" else None
        # MPS runs the full-vocabulary matmul about 3x slower than the same matmul in 8 row chunks, which give bitwise-equal logits.
        self.lm_head_chunks = base.lm_head.weight.chunk(8) if dev.type == "mps" else None

    def delta_prefill(self, q, k, v, g, beta, state, commit: bool) -> torch.Tensor:
        if self.device.type == "cuda":
            o, final = gated_delta_rule_chunk(q, k, v, g, beta, initial_state=state, output_final_state=True)
            if commit:
                state.copy_(final)
            return o
        steps = torch.full((q.shape[0],), q.shape[1], dtype=torch.long, device=q.device)
        return metal.gated_delta_rule_outputs(q, k, v, g, beta, state, steps, advance_state=commit)

    def delta_forward(self, q, k, v, g, beta, state) -> torch.Tensor:
        if self.device.type == "cuda":
            return delta_step(q, k, v, g, beta, state, False)[0]
        return metal.gated_delta_rule_outputs(q, k, v, g, beta, state, self.all_steps)

    def delta_commit(self, k, v, g, beta, state, keep, accepted) -> None:
        if self.device.type == "cuda":
            state.copy_(delta_step(torch.zeros_like(k), k, v, g * keep, beta * keep.to(beta.dtype), state, True)[1])
        else:
            metal.gated_delta_rule_advance_(k, v, g, beta, state, accepted)

    def vocab_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.lm_head_chunks is None:
            return self.base.lm_head(h)
        return torch.cat([F.linear(h, w) for w in self.lm_head_chunks], dim=-1)

    def bucket(self, n: int) -> int:
        b = 1
        while b < n:
            b *= 2
        return min(b, self.R)

    @torch.no_grad()
    def extend(self, seqs: list[list[int]], starts: list[int], commit: bool) -> torch.Tensor:
        dev, base = self.device, self.base
        B, T = len(seqs), max(len(s) for s in seqs)
        ids = torch.full((B, T), self.pad, dtype=torch.long)
        valid = torch.zeros(B, T, dtype=torch.bool)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            valid[b, :len(s)] = True
        ids, valid = ids.to(dev, non_blocking=True), valid.to(dev, non_blocking=True)
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        pos = torch.tensor(starts, device=dev)[:, None] + torch.arange(T, device=dev)[None]
        wpos = torch.where(valid, pos, self.trash)
        rows = torch.arange(B, device=dev)[:, None].expand(B, T)
        Lk = max(st + len(s) for st, s in zip(starts, seqs))
        mask = (self.ar_l[:Lk][None, None, :] <= pos[:, :, None])[:, None]
        cos, sin = base.model.rotary_emb(pos)
        keep = valid.float()[..., None]
        x = base.model.embed_tokens(ids)
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            if layer.layer_type == LINEAR:
                lin = layer.linear_attn
                ext = torch.cat((self.conv[i][:B], lin.in_proj_qkv(h).transpose(1, 2)), dim=-1)
                conv = F.silu(F.conv1d(ext, lin.conv1d.weight, groups=lin.conv_dim)[..., -T:]).transpose(1, 2)
                q, k, v = conv.split([lin.key_dim, lin.key_dim, lin.value_dim], dim=-1)
                rep = lin.num_v_heads // lin.num_k_heads
                q = q.reshape(B, T, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                k = k.reshape(B, T, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                v = v.reshape(B, T, lin.num_v_heads, lin.head_v_dim)
                beta = lin.in_proj_b(h).sigmoid() * keep.to(h.dtype)
                g = -lin.A_log.float().exp() * F.softplus(lin.in_proj_a(h).float() + lin.dt_bias) * keep
                o = self.delta_prefill(q, k, v, g, beta, self.rec[i][:B], commit)
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
                q, gate = att.q_proj(h).view(B, T, H, 2 * D).chunk(2, dim=-1)
                q = apply_rotary(att.q_norm(q), cos, sin)
                self.k[i][rows, wpos] = apply_rotary(att.k_norm(att.k_proj(h).view(B, T, Hkv, D)), cos, sin)
                self.v[i][rows, wpos] = att.v_proj(h).view(B, T, Hkv, D)
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
                hc, hm = h[:, :K], h[:, K:]
                ext = torch.cat((self.conv[i][:B], lin.in_proj_qkv(hc).transpose(1, 2)), dim=-1)
                conv = F.silu(F.conv1d(ext, lin.conv1d.weight, groups=lin.conv_dim)[..., -K:]).transpose(1, 2)
                q, k, v = conv.split([lin.key_dim, lin.key_dim, lin.value_dim], dim=-1)
                rep = lin.num_v_heads // lin.num_k_heads
                q = q.reshape(B, K, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                k = k.reshape(B, K, lin.num_k_heads, lin.head_k_dim).repeat_interleave(rep, dim=2)
                v = v.reshape(B, K, Hv, Dv)
                beta = lin.in_proj_b(hc).sigmoid()
                g = -lin.A_log.float().exp() * F.softplus(lin.in_proj_a(hc).float() + lin.dt_bias)
                o_c = self.delta_forward(q, k, v, g, beta, self.rec[i][:B])
                pending[i] = (k, v, g, beta, ext)
                self.dk[i][rows, pos_c] = apply_rotary(k, cos[:, :K], sin[:, :K])
                self.dv[i][rows, pos_c] = v
                qm = apply_rotary(view.delta_q[key](hm).view(B, M, Hv, Dv), cos[:, K:], sin[:, K:])
                km = apply_rotary(view.delta_k[key](hm).view(B, M, Hv, Dv), cos[:, K:], sin[:, K:])
                vm = view.delta_v[key](hm).view(B, M, Hv, Dv)
                self.dk[i][:B, Lw:Lw + M] = km
                self.dv[i][:B, Lw:Lw + M] = vm
                keys, vals = self.dk[i][:B, :Lw + M], self.dv[i][:B, :Lw + M]
                with sdpa_kernel(EFFICIENT):
                    o_m = F.scaled_dot_product_attention(qm.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2),
                                                         attn_mask=mask[:, :, K:], scale=Dv ** -0.5).transpose(1, 2)
                o = torch.cat((o_c, o_m), dim=1)
                z = lin.in_proj_z(h).view(B, K + M, Hv, Dv)
                o = lin.norm(o.reshape(-1, Dv), z.reshape(-1, Dv)).view(B, K + M, lin.value_dim)
                x = x + lin.out_proj(o)
            else:
                att = layer.self_attn
                H, Hkv, D = att.num_heads, att.num_kv_heads, att.head_dim
                hc, hm = h[:, :K], h[:, K:]
                qc, gc = att.q_proj(hc).view(B, K, H, 2 * D).chunk(2, dim=-1)
                qm, gm = view.attn_q[key](hm).view(B, M, H, 2 * D).chunk(2, dim=-1)
                q = apply_rotary(att.q_norm(torch.cat((qc, qm), 1)), cos, sin)
                kc = att.k_norm(att.k_proj(hc).view(B, K, Hkv, D))
                km = att.k_norm(view.attn_k[key](hm).view(B, M, Hkv, D))
                k = apply_rotary(torch.cat((kc, km), 1), cos, sin)
                self.k[i][rows, pos_c] = k[:, :K]
                self.v[i][rows, pos_c] = att.v_proj(hc).view(B, K, Hkv, D)
                self.k[i][:B, Lw:Lw + M] = k[:, K:]
                self.v[i][:B, Lw:Lw + M] = view.attn_v[key](hm).view(B, M, Hkv, D)
                keys, vals = self.k[i][:B, :Lw + M], self.v[i][:B, :Lw + M]
                with sdpa_kernel(EFFICIENT):
                    o = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2), attn_mask=mask,
                                                       scale=att.scaling, enable_gqa=True).transpose(1, 2)
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
        keep = (self.ar_k[None] < j[:, None]).float()[..., None]
        for i, (k, v, g, beta, ext) in pending.items():
            self.delta_commit(k, v, g, beta, self.rec[i][:B], keep, j)
            idx = (j[:, None] + torch.arange(ext.shape[-1] - K, device=self.device)[None])[:, None, :].expand(B, ext.shape[1], -1)
            self.conv[i][:B].copy_(ext.gather(2, idx))
        cand.copy_(torch.where(done[:, None], cand, nxt))
        done.logical_or_((first_eos < j) | (self.n_out[:B] + j >= self.cap[:B]))
        n.add_(j)
        self.n_out[:B].add_(j)

    def graph(self, B: int, Lw: int) -> torch.cuda.CUDAGraph:
        g = self.graphs.get((B, Lw))
        if g is not None:
            return g
        saved = self.done[:B].clone()
        self.done[:B].fill_(True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.cycle(B, Lw)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self.cycle(B, Lw)
        self.done[:B].copy_(saved)
        self.graphs[(B, Lw)] = g
        return g

    def cycle_runner(self, B: int, Lw: int) -> Callable[[], None]:
        if self.device.type == "cuda":
            return self.graph(B, Lw).replay
        return lambda: self.cycle(B, Lw)

    @torch.no_grad()
    def decode(self, B: int, prompts: list[list[int]], first: torch.Tensor, caps: list[int]) -> tuple[list[list[int]], list[bool]]:
        Bp = self.bucket(B)
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
        P = common_prefix(prompts)
        self.reset(self.bucket(B))
        if P > 0:
            self.extend([prompts[0][:P]], [0], commit=True)
            self.broadcast(B, P)
        h = self.extend([p[P:] for p in prompts], [P] * B, commit=True)
        last = h[torch.arange(B, device=self.device), torch.tensor([len(p) - P - 1 for p in prompts], device=self.device)]
        first = (self.vocab_logits(last) + self.bias).argmax(-1)
        budget = opts.max_think if opts.think else 0
        caps = [max(0, min(budget, self.L - len(p) - len(r) - slack)) for p, r in zip(prompts, rems)]
        nothink = None
        if opts.nothink_threshold is not None or not all(caps):
            seqs = [self.empty_think + r for r in rems]
            hn = self.extend(seqs, [len(p) for p in prompts], commit=False)
            nothink = [self.readout(hn[b], seqs[b]) for b in range(B)]
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
