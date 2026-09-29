from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.distributed as dist

from checkpoint import Checkpoint
from loader.dataloader import load_examples
from predictor import Predictor, PredictorConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a single softmax temperature for the pointer head on dev no-think rows and store it in the checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    checkpoint = Checkpoint.read(a.checkpoint)
    cfg = PredictorConfig.from_checkpoint(a.checkpoint, model=a.model)
    predictor = Predictor(cfg)
    examples, _ = load_examples(str(Path(a.data_dir) / f"{a.split}.jsonl"), predictor.encoder, predictor.sources, 8192)
    result = predictor.fit_temperature(examples)
    if predictor.rank == 0:
        checkpoint.metadata["temperature"] = predictor.head.temperature
        checkpoint.metadata_path.write_text(json.dumps(checkpoint.metadata, indent=2) + "\n")
        print(json.dumps({"checkpoint": a.checkpoint, **result}))
    if predictor.world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
