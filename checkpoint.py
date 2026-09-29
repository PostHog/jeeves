from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Checkpoint:
    path: Path
    metadata_path: Path
    metadata: dict

    @classmethod
    def read(cls, path: str | Path) -> Checkpoint:
        path = Path(path)
        if (path / "ckpt").is_dir():
            path = path / "ckpt"
        metadata_path = path / ("state.json" if (path / "state.json").exists() else "meta.json")
        return cls(path, metadata_path, json.loads(metadata_path.read_text()))

    def model_options(self, model: str | None = None) -> dict:
        saved = self.metadata.get("config", {})
        return {
            "model": model if model is not None else saved.get("model", "Qwen/Qwen3.5-9B"),
            "lora_r": saved.get("lora_r", 16),
            "lora_alpha": saved.get("lora_alpha", 32.0),
            "head_dim": saved.get("head_dim", 256),
        }

    def load_weights(self, model, head, device) -> None:
        import torch
        from safetensors.torch import load_file

        tensors = load_file(str(self.path / "lora.safetensors"))
        model.lora.load_state_dict({k.removeprefix("lora."): v for k, v in tensors.items()})
        head.load_state_dict(torch.load(self.path / "head.pt", map_location=device, weights_only=True))
        head.temperature = float(self.metadata.get("temperature", 1.0))

