from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

WANLI = "alisawuffles/WANLI"
WANLI_REVISION = "61c95318fd71c55b6ba355d76253254615f387ec"
NLI_OPTIONS = {"supported": "The evidence establishes the claim", "insufficient": "The evidence does not establish either",
               "contradicted": "The evidence establishes the opposite"}
NLI_LABELS = {"entailment": "supported", "neutral": "insufficient", "contradiction": "contradicted"}


def authored_records(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        keys = [o["id"] for o in r["options"]]
        criteria = {o["id"]: o["description"] for o in r["options"]}
        text = " ".join((r["state"] + " || " + r["question"]).casefold().split())
        out.append({"state": r["state"],
                    "questions": {"decision": {"type": "choice", "instructions": r["question"], "criteria": criteria,
                                               "label": keys[r["label"]], "src": f"semif_{r['family']}"}},
                    "_meta": {"source": "semif", "family": r["family"], "id": f"semif/{r['id']}", "group_id": f"semif/{r['group_id']}",
                              "row": r["id"], "variant": r["provenance"].get("variant", "original"), "partition": r["provenance"].get("partition"),
                              "repo": "TheoLeeCJ/SemIf-OpenJev", "revision": None, "split": "authored", "eval_only": True,
                              "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "row_sha256": hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()}})
    return out


def wanli_records(split: str, n: int, seed: Any, revision: str = WANLI_REVISION, max_chars: int = 4000, balanced: bool = True) -> list[dict[str, Any]]:
    ds = load_dataset(WANLI, split=split, revision=revision)
    rng = random.Random(f"{seed}:wanli:{split}")
    order = list(range(len(ds)))
    rng.shuffle(order)
    per_label = -(-n // 3) if balanced else n
    counts: dict[str, int] = {}
    out = []
    for i in order:
        ex = ds[i]
        gold = ex["gold"]
        if gold not in NLI_LABELS or len(ex["premise"]) + len(ex["hypothesis"]) > max_chars:
            continue
        if balanced and counts.get(gold, 0) >= per_label:
            continue
        counts[gold] = counts.get(gold, 0) + 1
        keys = list(NLI_OPTIONS)
        rng.shuffle(keys)
        text = " ".join(ex["premise"].casefold().split())
        out.append({"state": ex["premise"],
                    "questions": {"claim": {"type": "choice", "instructions": "Assess the claim using only the supplied evidence: " + ex["hypothesis"],
                                            "criteria": {k: NLI_OPTIONS[k] for k in keys}, "label": NLI_LABELS[gold], "src": "wanli"}},
                    "_meta": {"source": "wanli", "id": f"wanli/{split}/{ex['id']}", "group_id": f"wanli/{ex['pairID']}", "row": ex["id"],
                              "repo": WANLI, "revision": revision, "split": split, "variant": "clean",
                              "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "row_sha256": hashlib.sha256(json.dumps(ex, sort_keys=True).encode()).hexdigest()}})
        if len(out) == n:
            break
    return out
