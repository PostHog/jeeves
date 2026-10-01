from __future__ import annotations

import torch
import torch.nn as nn

from drafter.view import DrafterView
from inference.fp8_checkpoint import load_fp8_view
from inference.fp8_weights import FP8, quantize_row_blocks_on_cpu
from loader.dataloader import Encoder
from model import PointerHead, metal


class MetalFP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: torch.device | str | None = None):
        super().__init__()
        # MPS in torch 2.14 has no float8 dtype, so the codes are kept as bytes and the CPU makes and tiles them.
        self.register_buffer("tiled_codes", torch.empty(out_features, in_features, dtype=torch.uint8, device=device))
        self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False) if bias else None
        self.in_features, self.out_features = in_features, out_features

    @torch.no_grad()
    def load_codes(self, codes: torch.Tensor, scale: torch.Tensor) -> None:
        self.tiled_codes.copy_(metal.tile_fp8_codes(codes.view(torch.uint8)))
        self.scale.copy_(scale)

    @torch.no_grad()
    def load_weight(self, weight: torch.Tensor) -> None:
        for rows, codes, scale in quantize_row_blocks_on_cpu(weight):
            self.tiled_codes[rows] = metal.tile_fp8_codes(codes.view(torch.uint8)).to(self.tiled_codes.device)
            self.scale[rows] = scale.to(self.scale.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = metal.fp8_linear(x.to(torch.bfloat16), self.tiled_codes, self.scale)
        return y if self.bias is None else y + self.bias


def check_kernel_on_every_code(device: torch.device) -> None:
    # The kernel fills simdgroup matrices through thread_elements(), whose lane layout Apple does not document.
    codes = torch.arange(256, dtype=torch.uint8)
    finite = codes[~codes.view(FP8).float().isnan()]
    table = finite[(torch.arange(256)[:, None] + torch.arange(256)[None]) % len(finite)]
    picked = metal.fp8_linear(torch.eye(256, dtype=torch.bfloat16, device=device), metal.tile_fp8_codes(table).to(device),
                              torch.ones(256, device=device))
    if not torch.equal(picked.cpu().float(), table.view(FP8).float().t()):
        raise RuntimeError("fp8_linear does not decode e4m3fn codes correctly on this GPU; use precision='bf16'")


def load_view(model: str, drafter: str, block: int, device: torch.device) -> tuple[DrafterView, PointerHead, Encoder]:
    check_kernel_on_every_code(device)
    return load_fp8_view(model, drafter, block, device, MetalFP8Linear)
