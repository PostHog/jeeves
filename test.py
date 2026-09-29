from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.distributed as dist

from loader.dataloader import load_examples
from predictor import Predictor, PredictorConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on a split with the CISPO completion rule (torchrun for DDP).")
    ap.add_argument("checkpoint")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-think", type=int, default=2560)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gen-batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = PredictorConfig.from_checkpoint(a.checkpoint, model=a.model, eval_batch_size=a.batch_size,
                                          eval_think_limit=0, gen_batch_size=a.gen_batch_size,
                                          max_think=a.max_think, temperature=a.temperature,
                                          pad_multiple=a.pad_multiple, seed=a.seed)
    predictor = Predictor(cfg)
    examples, dropped = load_examples(str(Path(a.data_dir) / f"{a.split}.jsonl"), predictor.encoder, predictor.sources, a.max_seq_len,
                                      extra_len=a.max_think, limit=a.limit, seed=a.seed)
    result = predictor.evaluate(examples, think=True)
    result.update(split=a.split, checkpoint=a.checkpoint, examples=len(examples), dropped=dropped, max_think=a.max_think,
                  temperature=a.temperature)
    if predictor.rank == 0:
        print(json.dumps(result, indent=2))
        out = Path(a.out) if a.out else Path(a.checkpoint) / f"eval_{a.split}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
    if predictor.world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
