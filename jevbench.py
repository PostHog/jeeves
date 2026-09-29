from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from loader.dataloader import Example, build_examples, collate_rollouts, collate_sft
from prep.format import DataFormat, Question
from predictor import Predictor, PredictorConfig

TIER_FILES = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}
TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}
TIER_OPTION_COUNTS = {"easy": {2: 18, 4: 13, 5: 41}, "standard": {2: 32, 4: 40, 5: 12, 6: 12},
                      "judge": {2: 68, 9: 78}, "hard": {2: 77, 3: 26, 4: 73, 5: 38, 6: 6}}


def tier_chance(counts: dict[int, int]) -> float:
    n = sum(counts.values())
    return sum(c / k for k, c in counts.items()) / n


def to_record(task: dict, tier: str) -> DataFormat:
    q = task["question"]
    kind = q["type"]
    if kind == "choice":
        crit = q["criteria"]
        criteria = {k: crit.get(k) for k in task["labels"]}
        label = task["expected"]
    elif kind == "noul":
        criteria = q.get("criteria")
        label = task["expected"] == "yes"
    else:
        criteria = list(q["criteria"])
        label = int(task["expected"])
    question = Question(id="decision", type=kind, instructions=q["instructions"], criteria=criteria, label=label,
                        src=f"jevbench_{tier}_{task['family']}")
    meta = {"source": f"jevbench_{tier}", "id": task["id"], "group_id": task.get("group") or task["id"], "tier": tier,
            "family": task["family"], "n_labels": len(task["labels"]), "eval_only": True}
    return DataFormat(state=task["state"], questions=[question], meta=meta)


def load_tasks(root: Path) -> list[DataFormat]:
    out = []
    for tier, fname in TIER_FILES.items():
        for line in open(root / fname, encoding="utf-8"):
            if line.strip():
                out.append(to_record(json.loads(line), tier))
    return out


def ece_top_label(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    bins = [[0, 0.0, 0] for _ in range(n_bins)]
    for conf, correct in pairs:
        i = min(int(min(max(conf, 0.0), 1.0) * n_bins), n_bins - 1)
        bins[i][0] += 1
        bins[i][1] += conf
        bins[i][2] += int(correct)
    n = sum(b[0] for b in bins)
    return sum((b[0] / n) * abs(b[2] / b[0] - b[1] / b[0]) for b in bins if b[0])


@torch.no_grad()
def score_examples(predictor: Predictor, examples: list[Example], think: bool, batch_size: int) -> list[dict]:
    out = []
    with predictor.evaluation():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            t0 = time.time()
            if think:
                generated, _ = predictor.generate_chains(chunk)
                batch = collate_rollouts(chunk, generated, list(range(len(chunk))), predictor.m).to(predictor.device)
            else:
                batch = collate_sft(chunk, predictor.m).to(predictor.device)
            logits = predictor.scores(predictor.model.hidden_states(batch.input_ids, batch.attention_mask), batch)
            probs = F.softmax(logits, dim=-1)
            torch.cuda.synchronize()
            per_row = (time.time() - t0) / len(chunk)
            pred = probs.argmax(-1).tolist()
            pmax = probs.amax(-1).tolist()
            closed = batch.valid.tolist()
            for i, ex in enumerate(chunk):
                out.append({"id": ex.record_id, "pred": pred[i], "label": ex.label, "correct": pred[i] == ex.label, "confidence": pmax[i],
                            "closed": bool(closed[i]), "latency_s": per_row, "source": ex.source, "family": ex.question_id})
    return out


def summarize(rows: list[dict], tiers: dict[str, str], families: dict[str, str]) -> dict:
    by_tier: dict[str, list] = {}
    by_family: dict[str, list] = {}
    for r in rows:
        by_tier.setdefault(tiers[r["id"]], []).append(r)
        by_family.setdefault(f"{tiers[r['id']]}/{families[r['id']]}", []).append(r)
    acc = {t: sum(r["correct"] for r in rs) / len(rs) for t, rs in by_tier.items()}
    chance = {t: tier_chance(TIER_OPTION_COUNTS[t]) for t in acc}
    corrected = {t: max(0.0, 100 * (acc[t] - chance[t]) / (1 - chance[t])) for t in acc}
    w = sum(TIER_WEIGHTS[t] for t in acc)
    intelligence = sum(TIER_WEIGHTS[t] * corrected[t] for t in acc) / w
    pairs = [(r["confidence"], r["correct"]) for r in rows]
    ece = ece_top_label(pairs)
    lat = sorted(r["latency_s"] for r in rows)
    p50, p95 = lat[len(lat) // 2], lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    speed = lambda s: max(0.0, min(100.0, 100 - 20 * math.log10(max(s, 1e-6) / 0.1)))
    return {"n": len(rows), "accuracy_by_tier": {t: round(v, 4) for t, v in sorted(acc.items())},
            "chance_by_tier": {t: round(v, 4) for t, v in sorted(chance.items())},
            "intelligence_public": round(intelligence, 2), "ece": round(ece, 4), "calibration_ece_axis": round(max(0.0, 100 * (1 - ece / 0.5)), 2),
            "accuracy_all": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "closed_frac": round(sum(r["closed"] for r in rows) / len(rows), 4),
            "latency_p50_s": round(p50, 3), "latency_p95_s": round(p95, 3),
            "speed_axis_raw": round((speed(p50) + speed(p95)) / 2, 2),
            "accuracy_by_family": {k: round(sum(r["correct"] for r in rs) / len(rs), 4) for k, rs in sorted(by_family.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the JevBench public tasks (easy, standard, hard tiers) on a checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--tasks", default="data/jevbench")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-think", type=int, default=2560)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gen-batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=12288)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = PredictorConfig.from_checkpoint(a.checkpoint, model=a.model, gen_batch_size=a.gen_batch_size,
                                          max_think=a.max_think, temperature=a.temperature)
    predictor = Predictor(cfg)
    records = load_tasks(Path(a.tasks))
    tiers = {r.id: r.meta["tier"] for r in records}
    families = {r.id: r.meta["family"] for r in records}
    examples, dropped = build_examples(records, predictor.encoder, predictor.sources, a.max_seq_len, extra_len=a.max_think if not a.no_think else 0)
    shard = examples[predictor.rank::predictor.world]
    result = {"checkpoint": a.checkpoint, "tasks": len(records), "dropped_too_long": dropped, "max_think": a.max_think, "temperature": a.temperature}
    modes = [("empty", False)] + ([] if a.no_think else [("think", True)])
    for name, think in modes:
        rows = score_examples(predictor, shard, think, a.batch_size)
        if predictor.world > 1:
            gathered: list = [None] * predictor.world
            dist.all_gather_object(gathered, rows)
            rows = [r for part in gathered for r in part]
        if predictor.rank == 0:
            result[name] = summarize(rows, tiers, families)
            result[name]["rows"] = rows
    if predictor.rank == 0:
        ref_path = Path(a.tasks) / "reference.json"
        if ref_path.exists():
            result["reference_public_accuracy"] = json.loads(ref_path.read_text())["reference_public_accuracy"]
        out = Path(a.out) if a.out else Path(a.checkpoint) / "jevbench.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) and "rows" in v else v
                          for k, v in result.items()}, indent=2))
    if predictor.world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
