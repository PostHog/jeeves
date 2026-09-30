from __future__ import annotations

import torch
import torch.nn as nn

from model import metal

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


class MetalFP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        # MPS has no float8 dtype, and the CPU computes the scale and the cast with the same rounding as inference/fp8.py on CUDA.
        w = linear.weight.detach().cpu().float()
        scale = w.abs().amax(1).clamp(min=1e-12) / FP8_MAX
        device = linear.weight.device
        self.register_buffer("codes", (w / scale[:, None]).to(FP8).view(torch.uint8).to(device))
        self.register_buffer("scale", scale.to(device))
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().to(torch.bfloat16), requires_grad=False)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = metal.fp8_linear(x.to(torch.bfloat16), self.codes, self.scale)
        return y if self.bias is None else y + self.bias


def quantize(module: nn.Module) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features % 256 == 0 and child.out_features >= 1024 and child.out_features % 16 == 0:
            setattr(module, name, MetalFP8Linear(child))
            n += 1
        else:
            n += quantize(child)
    return n
