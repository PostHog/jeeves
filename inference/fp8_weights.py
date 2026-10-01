from __future__ import annotations

from typing import Callable, Iterator

import torch
import torch.nn as nn

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
# CUDA divides by a Python scalar as a multiply by its float32 reciprocal, so multiplying here gives the same scales on every device.
FP8_MAX_RECIPROCAL = float(torch.tensor(1 / FP8_MAX, dtype=torch.float32))
SCALE_SUFFIX = "_scale"
QUANTIZE_BLOCK_ROWS = 16384


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().float()
    scale = w.abs().amax(1).clamp(min=1e-12) * FP8_MAX_RECIPROCAL
    return (w / scale[:, None]).to(FP8).contiguous(), scale.contiguous()


def quantize_row_blocks_on_cpu(weight: torch.Tensor) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor]]:
    # Row blocks bound the float copies of the weight that quantize_rows makes.
    for start in range(0, weight.shape[0], QUANTIZE_BLOCK_ROWS):
        rows = slice(start, start + QUANTIZE_BLOCK_ROWS)
        yield rows, *quantize_rows(weight[rows].cpu())


def quantizable(linear: nn.Linear) -> bool:
    return linear.in_features % 256 == 0 and linear.out_features >= 1024 and linear.out_features % 16 == 0


def replace_linears(module: nn.Module, make: Callable[[nn.Linear], nn.Module]) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and quantizable(child):
            setattr(module, name, make(child))
            n += 1
        else:
            n += replace_linears(child, make)
    return n
