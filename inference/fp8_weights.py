from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
# CUDA divides by a Python scalar as a multiply by its float32 reciprocal, so multiplying here gives the same scales on every device.
FP8_MAX_RECIPROCAL = float(torch.tensor(1 / FP8_MAX, dtype=torch.float32))


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().float()
    scale = w.abs().amax(1).clamp(min=1e-12) * FP8_MAX_RECIPROCAL
    return (w / scale[:, None]).to(FP8).contiguous(), scale.contiguous()


def replace_linears(module: nn.Module, make: Callable[[nn.Linear], nn.Module]) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features % 256 == 0 and child.out_features >= 1024 and child.out_features % 16 == 0:
            setattr(module, name, make(child))
            n += 1
        else:
            n += replace_linears(child, make)
    return n
