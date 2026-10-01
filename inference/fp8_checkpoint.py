from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from drafter.view import DrafterView
from export import export_dir, load_head
from inference.fp8_weights import FP8, SCALE_SUFFIX, quantizable, quantize_row_blocks_on_cpu, replace_linears
from loader.dataloader import Encoder
from model import PointerHead, Qwen3_5_9BConfig, Qwen3_5ForCausalLM
from model.model import RotaryEmbedding, map_checkpoint_key

BASE = "base."


def meta_view(cfg: Qwen3_5_9BConfig, block: int = 4) -> DrafterView:
    with torch.device("meta"):
        return DrafterView(Qwen3_5ForCausalLM(cfg).to(torch.bfloat16), block=block)


def quantized_linears(view: DrafterView) -> set[str]:
    return {name for name, m in view.named_modules() if isinstance(m, nn.Linear) and quantizable(m)}


def quantize_on_cpu(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.empty(weight.shape, dtype=FP8)
    scale = torch.empty(weight.shape[0], dtype=torch.float32)
    for rows, block_codes, block_scale in quantize_row_blocks_on_cpu(weight):
        codes[rows], scale[rows] = block_codes, block_scale
    return codes, scale


def write_quantized(src: Path, dst: Path, linears: set[str]) -> dict[str, int]:
    tensors = {}
    with safe_open(str(src), framework="pt", device="cpu") as fh:
        for key in fh.keys():
            if key.endswith(".weight") and key.removesuffix(".weight") in linears:
                tensors[key], tensors[key + SCALE_SUFFIX] = quantize_on_cpu(fh.get_tensor(key))
            else:
                tensors[key] = fh.get_tensor(key)
    save_file(tensors, str(dst), metadata={"format": "pt"})
    return {key: t.numel() * t.element_size() for key, t in tensors.items()}


def checked(key: str, tensor: torch.Tensor, shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
    if tensor.shape != torch.Size(shape):
        raise ValueError(f"{key}: checkpoint {tuple(tensor.shape)} vs model {tuple(shape)}")
    return tensor


@torch.no_grad()
def load_tensors(module: nn.Module, files: list[Path], linear_class: type[nn.Module]) -> tuple[set[str], list[str]]:
    params = dict(module.named_parameters()) | dict(module.named_buffers())
    modules = dict(module.named_modules())
    loaded, unexpected = set(), []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
            scales, used_scales = {key for key in keys if key.endswith(SCALE_SUFFIX)}, set()
            for key in keys:
                name = map_checkpoint_key(key)
                if key in scales or name is None:
                    continue
                owner, _, leaf = name.rpartition(".")
                linear = modules.get(owner)
                if isinstance(linear, linear_class) and leaf == "weight":
                    weight = checked(key, fh.get_tensor(key), (linear.out_features, linear.in_features))
                    if key + SCALE_SUFFIX in scales:
                        linear.load_codes(weight, checked(key + SCALE_SUFFIX, fh.get_tensor(key + SCALE_SUFFIX), (linear.out_features,)))
                        used_scales.add(key + SCALE_SUFFIX)
                    else:
                        linear.load_weight(weight)
                    loaded.update(f"{owner}.{buffer}" for buffer, _ in linear.named_buffers())
                elif name in params:
                    params[name].copy_(checked(key, fh.get_tensor(key), params[name].shape))
                    loaded.add(name)
                else:
                    unexpected.append(key)
            unexpected += sorted(scales - used_scales)
    return loaded, unexpected


def load_fp8_view(path: str, drafter: str, block: int, device: torch.device,
                  linear_class: type[nn.Module]) -> tuple[DrafterView, PointerHead, Encoder]:
    path = export_dir(path)
    cfg = Qwen3_5_9BConfig.from_json(path / "config.json")
    view = meta_view(cfg, block)
    replace_linears(view, lambda linear: linear_class(linear.in_features, linear.out_features, bias=linear.bias is not None, device="meta"))
    view.to_empty(device=device)
    # Built on the device, as from_pretrained builds it, so the frequencies round the same way.
    with torch.device(device):
        view.base.model.rotary_emb = RotaryEmbedding(cfg)
    shards = sorted(set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values()))
    base_loaded, base_unexpected = load_tensors(view.base, [path / shard for shard in shards], linear_class)
    drafter_loaded, drafter_unexpected = load_tensors(view, [Path(drafter)], linear_class)
    loaded = {BASE + name for name in base_loaded} | drafter_loaded
    expected = {name for name, _ in [*view.named_parameters(), *view.named_buffers()] if not name.endswith("inv_freq")}
    missing = sorted(expected - loaded)
    if missing or base_unexpected or drafter_unexpected:
        raise ValueError(f"FP8 weights do not match the model: missing {missing[:3]}, unexpected {(base_unexpected + drafter_unexpected)[:3]}")
    view.requires_grad_(False)
    return view.eval(), load_head(path, cfg.hidden_size, device), Encoder(str(path))
