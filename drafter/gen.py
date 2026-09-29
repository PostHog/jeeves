from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from distributed import setup_distributed
from loader.dataloader import Encoder, collate_prompts, load_examples
from model import Qwen3_5ForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample on-policy thinking chains from a frozen checkpoint for drafter distillation.")
    ap.add_argument("--model", default="runs/fused")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="train,rl")
    ap.add_argument("--out", default="data/chains_402/chains.jsonl")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=400)
    a = ap.parse_args()
    rank, world, _ = setup_distributed()
    torch.manual_seed(a.seed + rank)
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = Encoder(a.tokenizer)
    examples = []
    for split in a.splits.split(","):
        ex, _ = load_examples(str(Path(a.data_dir) / f"{split}.jsonl"), encoder, {}, 1 << 30)
        examples.extend(ex)
    if a.limit:
        examples = examples[:a.limit]
    jobs = [(ex, s) for s in range(a.samples) for ex in examples][rank::world]
    jobs.sort(key=lambda j: len(j[0].prompt))
    model = Qwen3_5ForCausalLM.from_pretrained(a.model, device=device)
    model.eval()
    bias = torch.zeros(model.lm_head.weight.shape[0], device=device)
    bias[torch.tensor(encoder.banned_ids, device=device)] = -1e4
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = out.with_name(f"{out.stem}.rank{rank}{out.suffix}")
    eos = encoder.think_end_id
    t0 = time.time()
    n_tok = 0
    with open(path, "w", encoding="utf-8") as f, torch.no_grad():
        for start in range(0, len(jobs), a.batch_size):
            chunk = jobs[start:start + a.batch_size]
            ids, mask = collate_prompts([j[0] for j in chunk], encoder.markers)
            ids, mask = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            gen = model.generate_graphed(ids, attention_mask=mask, max_new_tokens=a.max_new, temperature=a.temperature, eos_token_id=eos,
                                         logit_bias=bias)
            gen = gen[:, ids.shape[1]:].cpu().tolist()
            for (ex, s), chain in zip(chunk, gen):
                if eos in chain:
                    chain = chain[:chain.index(eos) + 1]
                n_tok += len(chain)
                f.write(json.dumps({"id": f"{ex.record_id}/{ex.question_id}", "sample": s, "prompt": ex.prompt, "chain": chain}) + "\n")
            f.flush()
            if rank == 0:
                done = start + len(chunk)
                print(json.dumps({"done": done, "total": len(jobs), "tok_per_s": round(n_tok / (time.time() - t0)), "elapsed_s": round(time.time() - t0)}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
