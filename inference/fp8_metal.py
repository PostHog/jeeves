from __future__ import annotations

import torch
import torch.nn as nn

from inference.fp8_weights import FP8, quantize_rows, replace_linears
from model import metal

QUANTIZE_BLOCK_ROWS = 16384


class MetalFP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight, device = linear.weight.detach(), linear.weight.device
        self.register_buffer("codes", torch.empty(weight.shape, dtype=torch.uint8, device=device))
        self.register_buffer("scale", torch.empty(weight.shape[0], dtype=torch.float32, device=device))
        # MPS in torch 2.14 has no float8 dtype, so the CPU makes the codes, in row blocks to bound its float copies of the weight.
        for start in range(0, weight.shape[0], QUANTIZE_BLOCK_ROWS):
            codes, scale = quantize_rows(weight[start:start + QUANTIZE_BLOCK_ROWS].cpu())
            self.codes[start:start + len(scale)] = codes.view(torch.uint8).to(device)
            self.scale[start:start + len(scale)] = scale.to(device)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().to(torch.bfloat16), requires_grad=False)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = metal.fp8_linear(x.to(torch.bfloat16), self.codes, self.scale)
        return y if self.bias is None else y + self.bias


def check_kernel_on_every_code(device: torch.device) -> None:
    # The kernel fills simdgroup matrices through thread_elements(), whose lane layout Apple does not document.
    codes = torch.arange(256, dtype=torch.uint8)
    finite = codes[~codes.view(FP8).float().isnan()]
    table = finite[(torch.arange(256)[:, None] + torch.arange(256)[None]) % len(finite)]
    picked = metal.fp8_linear(torch.eye(256, dtype=torch.bfloat16, device=device), table.to(device), torch.ones(256, device=device))
    if not torch.equal(picked.cpu().float(), table.view(FP8).float().t()):
        raise RuntimeError("fp8_linear does not decode e4m3fn codes correctly on this GPU; use precision='bf16'")


def quantize(module: nn.Module) -> int:
    device = next(module.parameters()).device
    check_kernel_on_every_code(device)

    def replace(linear: nn.Linear) -> nn.Module:
        # Frees the bf16 weight of the layer replaced just before, so the allocator does not hold every old weight at once.
        torch.mps.empty_cache()
        return MetalFP8Linear(linear)

    return replace_linears(module, replace)
