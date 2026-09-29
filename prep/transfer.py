from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

from datasets import load_dataset

from .night2 import unknowable_records

MMLU_PRO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
BURIED_SOURCES = ("paws", "qnli", "tweet_offensive", "emotion")
BURIED_NOTE = "Answer about the primary record only; the other records are unrelated."


def mmlu_pro_records(n: int, seed: Any, revision: str = MMLU_PRO_REVISION) -> list[dict[str, Any]]:
    ds = load_dataset(MMLU_PRO, split="test", revision=revision)
    rng = random.Random(f"{seed}:mmlu_pro")
    out = []
    for i in rng.sample(range(len(ds)), min(len(ds), 4 * n)):
        ex = ds[i]
        options = [o for o in ex["options"] if o and o != "N/A"]
        if not 4 <= len(options) <= 10 or ex["answer_index"] >= len(options):
            continue
        keys = "abcdefghij"[:len(options)]
        text = " ".join(ex["question"].casefold().split())
        out.append({"state": {"category": ex["category"], "question": ex["question"]},
                    "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                                             "criteria": dict(zip(keys, options)), "label": keys[ex["answer_index"]], "src": "mmlu_pro"}},
                    "_meta": {"row": ex["question_id"], "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "source": "mmlu_pro",
                              "repo": MMLU_PRO, "revision": revision, "split": "test", "id": f"mmlu_pro/test/{ex['question_id']}",
                              "group_id": f"mmlu_pro/test/{ex['question_id']}", "variant": "clean", "eval_only": True,
                              "row_sha256": hashlib.sha256(json.dumps(ex, sort_keys=True, default=str).encode()).hexdigest()}})
        if len(out) == n:
            break
    return out


def buried_records(records: list[dict[str, Any]], n: int, seed: Any) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:buried")
    out = []
    pools = {s: [r for r in records if r["_meta"]["source"] == s and isinstance(r["state"], str)] for s in BURIED_SOURCES}
    per = -(-n // len(BURIED_SOURCES))
    for source, pool in pools.items():
        if len(pool) < 4:
            continue
        for r in rng.sample(pool, min(per, len(pool))):
            others = rng.sample([o for o in pool if o["_meta"]["id"] != r["_meta"]["id"]], 3)
            slot = rng.randrange(4)
            background = [o["state"] for o in others]
            rec = copy.deepcopy(r)
            rec["state"] = {"records": background[:slot] + [r["state"]] + background[slot:], "primary_record": slot + 1, "note": BURIED_NOTE}
            for q in rec["questions"].values():
                q["src"] = f"buried_{q['src']}"
            rec["_meta"] = {**r["_meta"], "source": "buried", "variant": "clean", "parent_id": r["_meta"]["id"], "parent_source": source,
                            "id": f"buried/{r['_meta']['id']}", "group_id": f"buried/{r['_meta']['id']}", "primary_slot": slot, "eval_only": True,
                            "text_sha256": hashlib.sha256(json.dumps(rec["state"], sort_keys=True).encode()).hexdigest()}
            out.append(rec)
    return out[:n]


def unknowable_eval_records(pairs_per_family: int, seed: Any) -> list[dict[str, Any]]:
    out = unknowable_records(pairs_per_family, seed, families=None)
    for r in out:
        r["_meta"]["source"] = r["_meta"]["source"].replace("night2_", "")
        r["_meta"]["id"] = r["_meta"]["id"].replace("night2_", "")
        r["_meta"]["group_id"] = r["_meta"]["group_id"].replace("night2_", "")
        if "control_id" in r["_meta"]:
            r["_meta"]["control_id"] = r["_meta"]["control_id"].replace("night2_", "")
        r["_meta"]["eval_only"] = True
    return out
