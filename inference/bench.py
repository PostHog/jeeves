from __future__ import annotations

import argparse
import json
import random
import time
from typing import TYPE_CHECKING

import torch

from inference.types import Options
from prep.format import read_jsonl

if TYPE_CHECKING:
    from inference.engine import Engine


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Accuracy and latency of the inference engine on a labelled split.")
    ap.add_argument("--model", default="runs/fused")
    ap.add_argument("--drafter", default="runs/drafter_k4/drafter.safetensors")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--data", default="data/dev.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--window-step", type=int, default=512)
    ap.add_argument("--max-think", type=int, default=2560)
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nothink-threshold", type=float, default=None)
    ap.add_argument("--compare-ar", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from inference.engine import Engine

    records = read_jsonl(a.data)
    if a.limit and a.limit < len(records):
        records = random.Random(a.seed).sample(records, a.limit)
    engine = Engine(a.model, a.drafter, block=a.block, precision=a.precision, max_rows=a.max_rows,
                    max_len=a.max_len, window_step=a.window_step)
    opts = Options(think=a.think, max_think=a.max_think, nothink_threshold=a.nothink_threshold)
    engine.answer(records[0], Options(max_think=16))
    rows, lat = [], []
    for rec in records:
        torch.accelerator.synchronize()
        t0 = time.time()
        results = engine.answer(rec, opts)
        torch.accelerator.synchronize()
        lat.append(time.time() - t0)
        for q, r in zip(rec.questions, results):
            p = r.probs
            nt = r.nothink_probs
            rows.append({"source": rec.source, "ok": int(max(range(len(p)), key=lambda i: p[i]) == q.label_index()),
                         "p_label": p[q.label_index()], "thought": r.thought, "closed": r.closed, "tokens": len(r.chain),
                         "nothink_ok": None if nt is None else int(max(range(len(nt)), key=lambda i: nt[i]) == q.label_index())})
    n = len(rows)
    lat_sorted = sorted(lat)
    thought = [r for r in rows if r["thought"]]
    summary = {"precision": engine.precision, "block": a.block, "records": len(records), "questions": n, "acc": sum(r["ok"] for r in rows) / n,
               "nll": -sum(torch.tensor(max(r["p_label"], 1e-9)).log().item() for r in rows) / n,
               "thought_frac": len(thought) / n, "closed_frac_of_thought": sum(r["closed"] for r in thought) / max(1, len(thought)),
               "mean_chain_tokens": sum(r["tokens"] for r in rows) / n,
               "latency_mean_s": sum(lat) / len(lat), "latency_p50_s": lat_sorted[len(lat) // 2],
               "latency_p90_s": lat_sorted[int(len(lat) * 0.9)],
               "chain_tok_per_s": sum(r["tokens"] for r in rows) / sum(lat)}
    if rows[0]["nothink_ok"] is not None:
        summary["nothink_acc"] = sum(r["nothink_ok"] for r in rows) / n
    if a.compare_ar:
        summary["ar_check"] = compare_ar(engine, records, a.compare_ar, a.max_think)
    print(json.dumps(summary, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps({"summary": summary, "rows": rows}) + "\n")


@torch.no_grad()
def compare_ar(engine: Engine, records, n: int, max_think: int) -> dict:
    enc = engine.encoder
    n = min(n, len(records))
    if n == 0:
        return {"n": 0, "exact_frac": 0.0, "mean_prefix_match": 0.0, "mean_ref_tokens": 0.0}
    exact, prefix, total = 0, 0, 0
    for rec in records[:n]:
        q = rec.questions[0]
        prompt = enc.encode(enc.prompt_text(rec, q))
        remainder = enc.encode(enc.remainder_text(q))
        cap = max(0, min(max_think, engine.L - len(prompt) - len(remainder) - engine.K - engine.J - 2))
        spec = engine.group(rec, [q], Options(max_think=max_think))[0].chain
        ids = torch.tensor(prompt, device=engine.device)[None]
        ref = engine.base.generate(ids, max_new_tokens=cap, temperature=0.0, eos_token_id=engine.eos,
                                           logit_bias=engine.bias.float())[0, len(prompt):].tolist()
        if engine.eos in ref:
            ref = ref[:ref.index(engine.eos) + 1]
        m = 0
        for x, y in zip(spec, ref):
            if x != y:
                break
            m += 1
        exact += int(spec == ref)
        prefix += m
        total += len(ref)
    return {"n": n, "exact_frac": exact / n, "mean_prefix_match": prefix / n, "mean_ref_tokens": total / n}


if __name__ == "__main__":
    main()
