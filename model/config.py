from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

LINEAR = "linear_attention"
FULL = "full_attention"


def hybrid_layer_types(num_layers: int, full_attention_interval: int = 4) -> tuple[str, ...]:
    return tuple(FULL if (i + 1) % full_attention_interval == 0 else LINEAR for i in range(num_layers))


@dataclass
class Qwen3_5_9BConfig:
    vocab_size: int = 248_320
    hidden_size: int = 4096
    intermediate_size: int = 12_288
    num_hidden_layers: int = 32
    layer_types: tuple[str, ...] = field(default_factory=lambda: hybrid_layer_types(32, 4))
    tie_word_embeddings: bool = False

    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0

    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262_144

    pad_token_id: int | None = None
    eos_token_id: int = 248_044
    bos_token_id: int | None = None

    dtype: str = "bfloat16"
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        self.layer_types = tuple(self.layer_types)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers={self.num_hidden_layers}"
            )
        bad = set(self.layer_types) - {LINEAR, FULL}
        if bad:
            raise ValueError(f"unknown layer types: {sorted(bad)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.hidden_act != "silu":
            raise ValueError("only silu / SwiGLU is implemented")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    def is_linear(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == LINEAR

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Qwen3_5_9BConfig":
        raw = dict(raw.get("text_config", raw))
        rope = raw.pop("rope_parameters", None) or raw.pop("rope_scaling", None) or {}
        if "rope_theta" in rope:
            raw["rope_theta"] = rope["rope_theta"]
        if "partial_rotary_factor" in rope:
            raw["partial_rotary_factor"] = rope["partial_rotary_factor"]
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope_type {rope['rope_type']!r}")
        interval = raw.pop("full_attention_interval", 4)
        if "layer_types" not in raw:
            raw["layer_types"] = hybrid_layer_types(raw.get("num_hidden_layers", 32), interval)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "Qwen3_5_9BConfig":
        path = os.fspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layer_types"] = list(self.layer_types)
        return d
