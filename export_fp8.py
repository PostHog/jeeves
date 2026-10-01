from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from export import export_dir
from inference.fp8_checkpoint import BASE, meta_view, quantized_linears, write_quantized
from model import Qwen3_5_9BConfig

NOT_COPIED = {"model.safetensors.index.json", "export.json", "README.md", ".gitattributes"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize an exported model and its drafters to the FP8 weights that --precision fp8 serves.")
    ap.add_argument("model", help="export directory or Hugging Face repo")
    ap.add_argument("--drafter", action="append", default=[], help="drafter safetensors to quantize into --out; repeat for more")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src, out = export_dir(a.model), Path(a.out)
    if len({Path(d).name for d in a.drafter}) < len(a.drafter):
        raise ValueError(f"drafters are written to --out under their file names, which must differ: {a.drafter}")
    if out.exists() and any(out.iterdir()) and not a.force:
        raise FileExistsError(f"{out} is not empty; pass --force")
    out.mkdir(parents=True, exist_ok=True)
    linears = quantized_linears(meta_view(Qwen3_5_9BConfig.from_json(src / "config.json")))
    base_linears = {name.removeprefix(BASE) for name in linears if name.startswith(BASE)}
    drafter_linears = {name for name in linears if not name.startswith(BASE)}
    weight_map, total = {}, 0
    for shard in sorted(set(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"].values())):
        sizes = write_quantized(src / shard, out / shard, base_linears)
        weight_map |= dict.fromkeys(sizes, shard)
        total += sum(sizes.values())
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
    for drafter in map(Path, a.drafter):
        total += sum(write_quantized(drafter, out / drafter.name, drafter_linears).values())
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in NOT_COPIED:
            shutil.copyfile(f, out / f.name)
    meta = json.loads((src / "export.json").read_text()) | {"precision": "fp8"}
    (out / "export.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(out), "quantized_linears": len(linears), "drafters": len(a.drafter), "total_gb": round(total / 2**30, 2)}))


if __name__ == "__main__":
    main()
