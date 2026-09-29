from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache

from inference.engine import Engine
from inference.types import Options
from prep.format import DataFormat, read_jsonl

RUNS = 6
MAX_ROWS, MAX_LEN = 4, 4096
EVAL_MAX_THINK = 256


def default_snapshot() -> str:
    config = try_to_load_from_cache("PostHog/jeeves", "config.json")
    if not isinstance(config, str):
        raise FileNotFoundError("PostHog/jeeves is not in the Hugging Face cache: run `hf download PostHog/jeeves`")
    return os.path.dirname(config)


def argmax(p: list[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def as_record(r, thought: bool | None = None) -> dict:
    return {"chain": r.chain, "probs": r.probs, "closed": r.closed, "thought": r.thought if thought is None else thought}


def workload(records: list[DataFormat]) -> list[tuple[DataFormat, Options]]:
    sample = random.Random(0).sample(records, 12)
    think = {2: 96, 4: 160, 8: 160}
    return [(rec, Options(max_think=think[i]) if i in think else Options(think=False)) for i, rec in enumerate(sample)]


def run(engine: Engine, jobs: list[tuple[DataFormat, Options]]) -> list[list[dict]]:
    return [[as_record(r) for r in engine.answer(rec, opts)] for rec, opts in jobs]


def compare(outputs: list[list[dict]], reference: list[list[dict]]) -> dict:
    pairs = [(a, b) for ra, rb in zip(outputs, reference, strict=True) for a, b in zip(ra, rb, strict=True)]
    thought = [(a, b) for a, b in pairs if b["thought"]]
    divergence = []
    for a, b in thought:
        if a["chain"] != b["chain"]:
            divergence.append(next((i for i, (x, y) in enumerate(zip(a["chain"], b["chain"])) if x != y), min(len(a["chain"]), len(b["chain"]))))
    return {"chains_identical": f"{len(thought) - len(divergence)}/{len(thought)}", "first_divergence": divergence,
            "max_prob_diff": max(abs(x - y) for a, b in pairs for x, y in zip(a["probs"], b["probs"], strict=True)),
            "argmax_agree": f"{sum(argmax(a['probs']) == argmax(b['probs']) for a, b in pairs)}/{len(pairs)}"}


def speed(engine: Engine, records: list[DataFormat]) -> tuple[dict, list[list[dict]]]:
    jobs = workload(records)
    times, outputs = [], []
    for _ in range(RUNS):
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        outputs.append(run(engine, jobs))
        torch.accelerator.synchronize()
        times.append(time.perf_counter() - t0)
    kept = sorted(times[1:])[1:-1]
    seconds = statistics.mean(kept)
    tokens = sum(len(q["chain"]) for req in outputs[0] for q in req)
    summary = {"seconds": round(seconds, 3), "runs": [round(t, 2) for t in times], "chain_tokens": tokens,
               "tok_per_s": round(tokens / seconds, 2), "deterministic": all(o == outputs[0] for o in outputs[1:])}
    return summary, outputs[0]


def evaluate(engine: Engine, records: list[DataFormat], n: int) -> tuple[dict, list[list[dict]]]:
    chosen = random.Random(1).sample(records, n)
    rows, outputs = [], []
    t0 = time.perf_counter()
    for rec in chosen:
        nothink = engine.answer(rec, Options(think=False))
        think = engine.answer(rec, Options(max_think=EVAL_MAX_THINK))
        outputs.append([as_record(r) for r in think] + [as_record(r, thought=False) for r in nothink])
        for q, t, nt in zip(rec.questions, think, nothink):
            label = q.label_index()
            rows.append((t.probs, nt.probs, label))
    summary = {"eval_questions": len(rows), "eval_seconds": round(time.perf_counter() - t0, 1),
               "think_acc": round(sum(argmax(t) == y for t, _, y in rows) / len(rows), 4),
               "nothink_acc": round(sum(argmax(nt) == y for _, nt, y in rows) / len(rows), 4),
               "think_nll": round(-sum(math.log(max(t[y], 1e-9)) for t, _, y in rows) / len(rows), 4),
               "nothink_nll": round(-sum(math.log(max(nt[y], 1e-9)) for _, nt, y in rows) / len(rows), 4)}
    return summary, outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed MPS speed and equivalence harness: 6 runs, drop the first, trimmed mean of the middle 3.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    model = a.model or default_snapshot()
    records = read_jsonl(a.data)
    engine = Engine(model, f"{model}/drafter_k4.safetensors", max_rows=MAX_ROWS, max_len=MAX_LEN)
    engine.answer(records[0], Options(max_think=16))
    if a.eval:
        summary, outputs = evaluate(engine, records, a.eval)
    else:
        summary, outputs = speed(engine, records)
    summary["driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 1) if engine.device.type == "mps" else None
    if a.reference:
        summary.update(compare(outputs, json.loads(Path(a.reference).read_text())))
    if a.save:
        Path(a.save).write_text(json.dumps(outputs))
    print("---")
    for k, v in summary.items():
        print(f"{k + ':':<20}{v}")


if __name__ == "__main__":
    main()
