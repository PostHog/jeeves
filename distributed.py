from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return rank, world, local


def all_reduce_(t: torch.Tensor, world: int, average: bool = False) -> torch.Tensor:
    if world > 1:
        dist.all_reduce(t)
        if average:
            t.div_(world)
    return t


def all_reduce_grads(params: list[torch.nn.Parameter], world: int) -> None:
    if world == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat.div_(world)
    offset = 0
    for g in grads:
        g.copy_(flat[offset:offset + g.numel()].view_as(g))
        offset += g.numel()
