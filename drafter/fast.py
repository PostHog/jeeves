from __future__ import annotations

import argparse
from pathlib import Path

from inference.bench import main as benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the serving decoder without FP8 and compare it with autoregressive decoding.")
    ap.add_argument("weights")
    ap.add_argument("--model", default="runs/fused")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--window-step", type=int, default=512)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    benchmark([
        "--model", a.model, "--drafter", a.weights, "--precision", "bf16", "--max-rows", "1",
        "--data", str(Path(a.data_dir) / "dev.jsonl"), "--limit", str(a.n), "--compare-ar", str(a.n),
        "--max-think", str(a.max_new), "--block", str(a.block),
        "--window-step", str(a.window_step), "--seed", str(a.seed),
    ])


if __name__ == "__main__":
    main()
