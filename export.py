from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from checkpoint import Checkpoint
from loader.dataloader import Encoder
from model import LoRAConfig, Qwen3_5ForCausalLM, Qwen3_5_9BConfig
from model.head import PointerHead
from model.model import resolve_checkpoint

SHARD_BYTES = 4 * 2**30


def load_fused(checkpoint: str, model_name: str | None, device: str, dtype: torch.dtype) -> tuple[Qwen3_5ForCausalLM, PointerHead, dict]:
    checkpoint = Checkpoint.read(checkpoint)
    options = checkpoint.model_options(model_name)
    lora = LoRAConfig(r=options["lora_r"], alpha=options["lora_alpha"])
    model = Qwen3_5ForCausalLM.from_pretrained(options["model"], device=device, dtype=dtype, lora=lora)
    head = PointerHead(model.cfg.hidden_size, options["head_dim"]).to(device)
    checkpoint.load_weights(model, head, device)
    model.remove_lora(merge=True)
    head.eval()
    return model, head, checkpoint.metadata


def write_shards(tensors: dict[str, torch.Tensor], out: Path) -> dict:
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name, t in tensors.items():
        nbytes = t.numel() * t.element_size()
        if size + nbytes > SHARD_BYTES and shards[-1]:
            shards.append({})
            size = 0
        shards[-1][name] = t
        size += nbytes
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: v.contiguous().cpu() for k, v in shard.items()}, str(out / fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    return {"metadata": {"total_size": total}, "weight_map": weight_map}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse a LoRA checkpoint into Qwen3.5 and bundle it with the pointer head.")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.force:
        raise FileExistsError(f"{out} is not empty; pass --force")
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, a.dtype)
    model, head, state = load_fused(a.checkpoint, a.model, a.device, dtype)
    tensors = {k: v.detach() for k, v in model.state_dict().items() if not k.endswith("inv_freq")}
    index = write_shards(tensors, out)
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, out / "head.pt")
    (out / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2) + "\n")
    model_name = Checkpoint.read(a.checkpoint).model_options(a.model)["model"]
    src = Path(resolve_checkpoint(model_name))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "generation_config.json"):
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
    (out / "export.json").write_text(json.dumps({
        "source_checkpoint": a.checkpoint, "base_model": model_name, "dtype": a.dtype, "format": Encoder.FORMAT,
        "head_dim": head.q.out_features, "temperature": head.temperature, "stage": state.get("stage"), "step": state.get("step"), "sources": state.get("sources"),
        "metrics": state.get("metrics"), "training_config": state.get("config"),
    }, indent=2) + "\n")
    print(json.dumps({"out": str(out), "tensors": len(tensors), "shards": len(set(index["weight_map"].values())),
                      "total_gb": round(index["metadata"]["total_size"] / 2**30, 2)}))


def export_dir(path: str) -> Path:
    return Path(path) if Path(path).is_dir() else Path(snapshot_download(path))


def load_head(path: Path, hidden_size: int, device: str | torch.device) -> PointerHead:
    meta = json.loads((path / "export.json").read_text())
    head = PointerHead(hidden_size, meta["head_dim"]).to(device)
    head.load_state_dict(torch.load(path / "head.pt", map_location=device))
    head.temperature = float(meta.get("temperature", 1.0))
    return head.eval()


def load_export(path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> tuple[Qwen3_5ForCausalLM, PointerHead, Encoder]:
    path = export_dir(path)
    precision = json.loads((path / "export.json").read_text()).get("precision", "bf16")
    if precision != "bf16":
        raise ValueError(f"{path} stores {precision} weights; serve it with precision={precision!r}")
    model = Qwen3_5ForCausalLM.from_pretrained(str(path), cfg=Qwen3_5_9BConfig.from_json(path / "config.json"), device=device, dtype=dtype)
    return model, load_head(path, model.cfg.hidden_size, device), Encoder(str(path))


if __name__ == "__main__":
    main()
