from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    dtype: torch.dtype = torch.float32
    rslora: bool = False

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.targets = tuple(self.targets)

    @property
    def scaling(self) -> float:
        return self.alpha / math.sqrt(self.r) if self.rslora else self.alpha / self.r


class _CachedCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param: torch.Tensor, cached: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return cached

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        g = grad.to(torch.float32)
        return (g * ctx.scale if ctx.scale != 1.0 else g), None, None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = cfg.r
        self.scaling = cfg.scaling
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(cfg.r, base.in_features, device=dev, dtype=cfg.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.r, device=dev, dtype=cfg.dtype))
        self.reset_lora_parameters()

        self.merged = False
        self.enabled = True
        self.cache_dtype: torch.dtype | None = None
        self.cached_A: torch.Tensor | None = None
        self.cached_B: torch.Tensor | None = None

    @torch.no_grad()
    def refresh_cache(self, dtype: torch.dtype) -> None:
        if self.cached_A is None or self.cache_dtype != dtype:
            self.cached_A = torch.empty_like(self.lora_A, dtype=dtype)
            self.cached_B = torch.empty_like(self.lora_B, dtype=dtype)
            self.cache_dtype = dtype
        self.cached_A.copy_(self.lora_A)
        self.cached_B.copy_(self.lora_B * self.scaling)

    def drop_cache(self) -> None:
        self.cached_A = self.cached_B = self.cache_dtype = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @torch.no_grad()
    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight.add_(self.delta_weight())
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.delta_weight())
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base.weight, self.base.bias)
        if self.enabled and not self.merged:
            h = self.dropout(x)
            if self.cached_A is not None and self.cache_dtype == h.dtype:
                a = _CachedCast.apply(self.lora_A, self.cached_A, 1.0)
                b = _CachedCast.apply(self.lora_B, self.cached_B, self.scaling)
                y = F.linear(F.linear(h, a), b).add_(y)
            else:
                h = F.linear(h, self.lora_A.to(h.dtype))
                y = y + F.linear(h, self.lora_B.to(h.dtype)) * self.scaling
        return y

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, "
                f"scaling={self.scaling:g}, merged={self.merged}, enabled={self.enabled}")


def iter_lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            yield name, mod


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> "OrderedDict[str, LoRALinear]":
    replaced: "OrderedDict[str, LoRALinear]" = OrderedDict()
    targets = set(cfg.targets)
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in targets]
    if not candidates:
        raise ValueError(f"no nn.Linear matched LoRA targets {sorted(targets)}")
    for name, linear in candidates:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = LoRALinear(linear, cfg)
        setattr(parent, attr, wrapped)
        replaced[name] = wrapped
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for _, mod in iter_lora_modules(model):
        mod.lora_A.requires_grad_(True)
        mod.lora_B.requires_grad_(True)


class LoRAAdapter:
    def __init__(self, modules: "OrderedDict[str, LoRALinear]", cfg: LoRAConfig):
        self.cfg = cfg
        self._modules: "OrderedDict[str, LoRALinear]" = OrderedDict(modules)

    @classmethod
    def from_model(cls, model: nn.Module, cfg: LoRAConfig) -> "LoRAAdapter":
        return cls(OrderedDict(iter_lora_modules(model)), cfg)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __getitem__(self, name: str) -> LoRALinear:
        return self._modules[name]

    def modules(self) -> Iterable[tuple[str, LoRALinear]]:
        return self._modules.items()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, mod in self._modules.items():
            yield f"{name}.lora_A", mod.lora_A
            yield f"{name}.lora_B", mod.lora_B

    def parameters(self) -> Iterator[nn.Parameter]:
        for _, p in self.named_parameters():
            yield p

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def requires_grad_(self, flag: bool = True) -> "LoRAAdapter":
        for p in self.parameters():
            p.requires_grad_(flag)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters():
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict((k, p.detach()) for k, p in self.named_parameters())

    @torch.no_grad()
    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        missing, unexpected = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
        if strict and (missing or unexpected):
            raise KeyError(f"LoRA state mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
        for k, v in sd.items():
            if k in own:
                own[k].copy_(v.to(own[k].dtype))

    def save(self, path: str) -> None:
        meta = {"r": str(self.cfg.r), "alpha": str(self.cfg.alpha), "targets": ",".join(self.cfg.targets),
                "rslora": str(self.cfg.rslora)}
        save_file({k: v.contiguous().cpu() for k, v in self.state_dict().items()}, path, metadata=meta)

    def load(self, path: str, strict: bool = True) -> None:
        self.load_state_dict(load_file(path), strict=strict)

    def merge(self) -> None:
        for m in self._modules.values():
            m.merge()

    def unmerge(self) -> None:
        for m in self._modules.values():
            m.unmerge()

    def enable(self) -> None:
        for m in self._modules.values():
            m.enabled = True

    def disable(self) -> None:
        for m in self._modules.values():
            m.enabled = False

    @torch.no_grad()
    def reset(self) -> None:
        for m in self._modules.values():
            m.reset_lora_parameters()

    def refresh_cache(self, dtype: torch.dtype) -> None:
        for m in self._modules.values():
            m.refresh_cache(dtype)

    def drop_cache(self) -> None:
        for m in self._modules.values():
            m.drop_cache()

    def __repr__(self) -> str:
        return (f"LoRAAdapter(modules={len(self)}, r={self.cfg.r}, alpha={self.cfg.alpha}, "
                f"params={self.num_parameters:,}, targets={list(self.cfg.targets)})")
