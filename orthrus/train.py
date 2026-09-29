from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from loader.dataloader import Encoder
from metrics import Metrics
from model import Qwen3_5ForCausalLM
from orthrus.data import Batches, load_streams, pack
from orthrus.view import OrthrusView, build_anchor_layout
from distributed import all_reduce_grads, setup_distributed


@dataclass
class OrthrusConfig:
    run_dir: str = "runs/orthrus_k8"
    model: str = "runs/fused"
    tokenizer: str = "Qwen/Qwen3.5-9B"
    chains: str = "data/chains_402/chains.rank*.jsonl"
    init: str = ""
    length: int = 2048
    block: int = 8
    blocks: int = 512
    batch_size: int = 2
    lr: float = 1.5e-4
    warmup: int = 100
    steps: int = 2500
    log_every: int = 100
    limit: int = 0
    seed: int = 0
    compile: bool = False
    grad_checkpoint: bool = True


def train_orthrus(a: OrthrusConfig) -> None:
    rank, world, _ = setup_distributed()
    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = Encoder(a.tokenizer)
    base = Qwen3_5ForCausalLM.from_pretrained(a.model, device=device)
    view = OrthrusView(base, block=a.block).to(device)
    view.grad_checkpoint = a.grad_checkpoint
    if a.init:
        view.load_state_dict({k: v.to(device) for k, v in load_file(a.init).items()}, strict=False)
    if a.compile:
        view.compile_layers()
    params = view.trainable_parameters()
    view.refresh_cache()
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    streams = load_streams(a.chains, encoder, limit=a.limit, seed=a.seed)
    packed = pack(streams, a.length, encoder.tok.eos_token_id)
    packed = packed[rank::world]
    batches = Batches(packed, a.batch_size, seed=a.seed + rank)
    out = Path(a.run_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a") if rank == 0 else None
    gen = torch.Generator().manual_seed(a.seed + rank)
    if rank == 0:
        print(json.dumps({"event": "setup", "world": world, "streams": len(streams), "packed_rows_per_rank": int(packed.shape[0]),
                          "tokens_per_rank": int(packed.numel()), "trainable_params": n_params,
                          "base_params": sum(p.numel() for p in base.parameters()), "mask_positions_per_row": a.blocks * (a.block - 1),
                          "rows_per_step": a.batch_size * world}), flush=True)
    step = 0
    acc = Metrics(("kl", "top1_agree", "grad_norm"), device)
    t0 = time.time()
    tokens_seen = 0
    while step < a.steps:
        for ids in batches:
            if step >= a.steps:
                break
            ids = ids.to(device, non_blocking=True)
            mask_pos, attn_mask = build_anchor_layout(a.length, a.block, a.blocks, gen, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_full, attn_kv = view.clean_pass(ids)
                teacher_h = teacher_full.index_select(1, mask_pos)
                student_h = view.mask_pass(attn_kv, mask_pos, attn_mask, ids.shape[0])
                loss, agree = view.kl_loss(student_h, teacher_h)
            loss.backward()
            all_reduce_grads(params, world)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            view.refresh_cache()
            step += 1
            tokens_seen += ids.numel()
            acc.add(kl=loss, top1_agree=agree, grad_norm=norm)
            if step % a.log_every == 0:
                torch.cuda.synchronize()
                dt = time.time() - t0
                vals = acc.means(world)
                rec = {"step": step, "kl": round(vals["kl"], 4), "top1_agree": round(vals["top1_agree"], 4), "grad_norm": round(vals["grad_norm"], 3),
                       "lr": sched.get_last_lr()[0], "s_per_step": round(dt / acc.count, 3),
                       "tokens_per_s": round(tokens_seen * world / dt), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                if rank == 0:
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                acc.reset()
                t0 = time.time()
                tokens_seen = 0
    if rank == 0:
        save_file({k: v.detach().contiguous().cpu() for k, v in view.state_dict().items() if not k.startswith("base.")}, str(out / "orthrus.safetensors"))
        (out / "config.json").write_text(json.dumps(asdict(a), indent=2) + "\n")
        print("saved", out / "orthrus.safetensors")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
