from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prep import composition, contrastive, night2, public, semif, transfer
from prep.format import DataFormat, option_text, render, state_hash, write_jsonl

SPLITS = ("train", "rl", "dev", "test")
RL_FRACTION = {
    "wanli": 0.4, "semif": 0.4,
    "contrastive": 0.4, "compositional": 0.4, "night2_dates": 0.4, "mnli": 0.4, "boolq": 0.4,
    "agnews": 0.15, "banking77": 0.15, "dbpedia14": 0.15, "trec": 0.15, "sst5": 0.15, "yelp": 0.15, "amazon": 0.15,
    "imdb": 0.15, "night2_assertion": 0.15,
    "night2_unknowable": 0.0, "night2_unknowable_control": 0.0,
}
MAX_STATE, MAX_BRANCH, MAX_PACKED, HEADROOM = 384, 1024, 2048, 64
NONE_KEY = "none_of_these"
NONE_TEXT = "None of these options describes the answer"


def source_seed(seed: Any, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


class Admission:
    def __init__(self, tokenizer_repo: str):
        self.tok = Tokenizer.from_file(hf_hub_download(tokenizer_repo, "tokenizer.json"))

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def fits(self, record: dict[str, Any], headroom: int = 0) -> bool:
        state_len = 1 + self.count(render(record["state"]))
        if state_len > MAX_STATE:
            return False
        total = state_len
        for q in record["questions"].values():
            if q["type"] == "choice":
                options = [option_text(k, v) for k, v in q["criteria"].items()]
            elif q["type"] == "noul":
                c = q.get("criteria") or {}
                options = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            else:
                options = [render(x) for x in q["criteria"]]
            branch = 2 + self.count(render(q["instructions"])) + sum(2 + self.count(o) for o in options)
            if branch > MAX_BRANCH - headroom - state_len:
                return False
            total += branch
        return total <= MAX_PACKED


def select_unique(records: list[dict], count: int, seen: set[str], admission: Admission, report: Counter) -> list[dict]:
    selected = []
    for record in sorted(records, key=lambda r: r["_meta"]["row_sha256"]):
        report["considered"] += 1
        key = record["_meta"]["text_sha256"]
        if key in seen:
            report["duplicate_state"] += 1
            continue
        if not admission.fits(record, HEADROOM):
            report["context_rejected"] += 1
            continue
        record["_meta"].update(group_id=record["_meta"]["id"], variant="clean")
        seen.add(key)
        selected.append(record)
        report["accepted"] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)}/{count} records fit the context policy")


def variant_copy(record: dict, variant: str) -> dict:
    r = copy.deepcopy(record)
    r["_meta"]["group_id"] = record["_meta"].get("group_id", record["_meta"]["id"])
    r["_meta"]["parent_id"] = record["_meta"]["id"]
    r["_meta"]["id"] += "/" + variant
    r["_meta"]["variant"] = variant
    return r


def contrast_cases(record: dict, seed: Any) -> list[dict]:
    candidates = [(qid, q) for qid, q in record["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3]
    if not candidates:
        return []
    qid, q = candidates[0]
    if NONE_KEY in q["criteria"]:
        raise ValueError("reserved contrast option collision")
    out = []
    for variant in ("none_present", "none_absent"):
        r = variant_copy(record, variant)
        rq = copy.deepcopy(q)
        rq["criteria"][NONE_KEY] = NONE_TEXT
        removed = None
        if variant == "none_absent":
            removed = rq["label"]
            rq["criteria"].pop(removed)
            rq["label"] = NONE_KEY
        if rq.get("target") is not None:
            # A target is keyed by option name, so the one this record inherited stops describing
            # the variant the moment the option set changes: for none_absent the removed option's
            # mass has nowhere to go and none_of_these - now the declared answer - carries none,
            # which renormalises the target onto the distractors. Rebuild it over the new options
            # and hand the removed option's mass to the key that replaced it. For none_present
            # nothing was removed and none_of_these is a distractor, so it takes no mass.
            old = rq["target"]
            rq["target"] = {k: float(old.get(k, 0.0)) for k in rq["criteria"]}
            if removed is not None:
                rq["target"][NONE_KEY] += float(old.get(removed, 0.0))
        keys = list(rq["criteria"])
        random.Random(source_seed(seed, record["_meta"]["id"])).shuffle(keys)
        rq["criteria"] = {k: rq["criteria"][k] for k in keys}
        r["questions"] = {qid: rq}
        r["_meta"]["none_key"] = NONE_KEY
        out.append(r)
    r = variant_copy(record, "permuted")
    rng = random.Random(source_seed(seed, record["_meta"]["id"]))
    for rq in r["questions"].values():
        if rq["type"] == "choice":
            keys = list(rq["criteria"])
            rng.shuffle(keys)
            rq["criteria"] = {k: rq["criteria"][k] for k in keys}
    out.append(r)
    return out


def resolve_revisions(sources: list[str]) -> dict[str, str | None]:
    api = HfApi()
    out: dict[str, str | None] = {}
    for name in sources:
        repo = public.REPOS[name].partition(":")[0]
        if repo in out:
            continue
        if repo in public.PINNED_REVISIONS:
            out[repo] = public.PINNED_REVISIONS[repo]
            continue
        try:
            out[repo] = api.dataset_info(repo, revision=public.PARQUET_BRANCH.get(repo)).sha
        except Exception:
            out[repo] = None
    return out


def semantic_hash(record: dict) -> str:
    state = record["state"]
    if isinstance(state, dict) and "policy" in state and "case" in state:
        state = {"policy": state["policy"], "sentences": sorted(x.rstrip(".") for x in state["case"].split(". "))}
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def unique_pairs(n_pairs: int, seed: Any, families, seen: set[str], admission: Admission, label: str) -> list[dict]:
    candidates = contrastive.generate(3 * n_pairs, seed, families)
    out, counts = [], Counter()
    for a, b in zip(candidates[::2], candidates[1::2]):
        family = a["_meta"]["family"]
        hashes = {semantic_hash(a), semantic_hash(b)}
        if counts[family] >= n_pairs or hashes & seen or len(hashes) < 2:
            continue
        for r in (a, b):
            if not admission.fits(r):
                raise ValueError(f"{label}: synthetic record exceeds the context")
            r["_meta"].update(group_id=r["_meta"]["family_id"], text_sha256=semantic_hash(r))
        out += [a, b]
        seen.update(hashes)
        counts[family] += 1
    short = {f: counts[f] for f in families if counts[f] < n_pairs}
    if short:
        raise ValueError(f"{label}: insufficient unique pairs {short}; change the seed")
    return out


def admit_synthetic(records: list[dict], reserved: set[str], admission: Admission, label: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["_meta"]["group_id"], []).append(r)
    for group in groups.values():
        if group[0]["_meta"]["source"] in ("compositional", "composition_holdout"):
            composition.check_group(group)
        hashes = {r["_meta"]["text_sha256"] for r in group}
        if hashes & reserved:
            raise ValueError(f"{label}: synthetic state collides with an existing state; change the seed")
        for r in group:
            if not admission.fits(r):
                raise ValueError(f"{label}: synthetic record exceeds the context")
        reserved.update(hashes)
    return records


SFT_BUDGET = {
    "agnews": 1000, "yelp": 1000, "banking77": 1000, "boolq": 1000, "mnli": 1000, "sst5": 1000, "trec": 1000, "dbpedia14": 1000,
    "amazon": 1000, "imdb": 1000, "wanli": 1000, "contrastive": 896, "compositional": 1680, "night2_dates": 900,
    "night2_assertion": 800, "night2_unknowable": 270, "night2_unknowable_control": 270, "semif": 72,
}


def group_hash(seed: Any, group: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{seed}:rl:{group}".encode()).digest()[:8], "big") / 2**64


def reserve_rl(records: list[dict], seed: Any, fractions: dict[str, float], rl_target: int) -> tuple[list[dict], list[dict]]:
    by_source: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        by_source.setdefault(r["_meta"]["source"], {}).setdefault(r["_meta"]["group_id"], []).append(r)
    train, remainder = [], {}
    for source, groups in by_source.items():
        budget = SFT_BUDGET.get(source, len(records))
        taken = 0
        for gid in sorted(groups, key=lambda g: group_hash(seed, g)):
            if taken < budget:
                train.extend(groups[gid])
                taken += len(groups[gid])
            else:
                remainder.setdefault(source, []).append(groups[gid])
    eligible = {s: sorted(gs, key=lambda g: group_hash(seed, g[0]["_meta"]["group_id"])) for s, gs in remainder.items() if fractions.get(s, 0.0) > 0}
    available = {s: sum(len(r["questions"]) for g in gs for r in g) for s, gs in eligible.items()}
    quota: dict[str, int] = {}
    left, pool = rl_target, dict(available)
    while pool and left > 0:
        share = left // len(pool)
        capped = {s: n for s, n in pool.items() if n <= share}
        if not capped:
            quota.update({s: quota.get(s, 0) + share for s in pool})
            break
        for s, n in capped.items():
            quota[s] = quota.get(s, 0) + n
            left -= n
            pool.pop(s)
    rl = []
    for source, gs in eligible.items():
        got = 0
        for g in gs:
            if got >= quota.get(source, 0):
                break
            rl.extend(g)
            got += sum(len(r["questions"]) for r in g)
    for r in rl:
        r["_meta"]["stage"] = "rl"
    for r in train:
        r["_meta"]["stage"] = "sft"
    return train, rl


def validate(records: list[dict]) -> None:
    for r in records:
        DataFormat.from_dict(r)
        for q in DataFormat.from_dict(r).questions:
            q.label_index()
            q.options()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download public datasets and generate policy data in kev's labelled-request form.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--seed", default="20260918")
    ap.add_argument("--train", type=int, default=4500)
    ap.add_argument("--dev", type=int, default=80)
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--variants-per-source", type=int, default=12)
    ap.add_argument("--contrastive-pairs", type=int, default=448)
    ap.add_argument("--contrastive-eval-pairs", type=int, default=16)
    ap.add_argument("--structures", type=int, default=60)
    ap.add_argument("--groups-per-structure", type=int, default=28)
    ap.add_argument("--dev-shape-groups", type=int, default=4)
    ap.add_argument("--holdout-shape-groups", type=int, default=8)
    ap.add_argument("--night2-date-pairs", type=int, default=600)
    ap.add_argument("--night2-unknowable-pairs", type=int, default=60)
    ap.add_argument("--night2-assertions", type=int, default=3200)
    ap.add_argument("--no-night2", action="store_true")
    ap.add_argument("--mmlu-pro", type=int, default=200)
    ap.add_argument("--buried", type=int, default=80)
    ap.add_argument("--unknowable-eval-pairs", type=int, default=5)
    ap.add_argument("--semif-tasks", default=str(Path(__file__).resolve().parents[1] / "data" / "semif" / "authored144.jsonl"))
    ap.add_argument("--wanli-eval", type=int, default=256)
    ap.add_argument("--wanli-train", type=int, default=4500)
    ap.add_argument("--rl-fraction", type=float, default=None)
    ap.add_argument("--rl-target", type=int, default=10000)
    ap.add_argument("--sources", default=",".join(public.TRAINABLE + public.EVAL_ONLY))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.smoke:
        a.train, a.dev, a.test, a.variants_per_source = 12, 6, 6, 2
        a.contrastive_pairs, a.contrastive_eval_pairs = 4, 2
        a.structures, a.groups_per_structure, a.dev_shape_groups, a.holdout_shape_groups = 4, 2, 1, 1
        a.night2_date_pairs, a.night2_unknowable_pairs, a.night2_assertions = 3, 2, 20
        a.mmlu_pro, a.buried, a.unknowable_eval_pairs = 8, 8, 1
        a.wanli_eval, a.wanli_train = 12, 12

    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.force:
        raise FileExistsError(f"{out} is not empty; pass --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    sources = [s for s in a.sources.split(",") if s]
    unknown = set(sources) - set(public.SOURCES)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    admission = Admission(a.tokenizer)
    revisions = resolve_revisions(sources)
    parts: dict[str, list[dict]] = {s: [] for s in SPLITS}
    seen: set[str] = set()
    manifest: dict[str, Any] = {"seed": a.seed, "tokenizer": a.tokenizer,
                                "context": {"max_state": MAX_STATE, "max_branch": MAX_BRANCH, "max_packed": MAX_PACKED,
                                            "admission_headroom": HEADROOM},
                                "trainable_sources": [s for s in sources if s in public.TRAINABLE],
                                "eval_only_sources": [s for s in sources if s in public.EVAL_ONLY],
                                "dataset_revisions": revisions, "admission": {}, "files": {}}

    for source in sources:
        report: Counter = Counter()
        repo = public.REPOS[source].partition(":")[0]
        eval_pool = public.build(source, "eval", max(3 * (a.dev + a.test), 600 if not a.smoke else 60), a.seed, revisions[repo])
        chosen = select_unique(eval_pool, a.dev + a.test, seen, admission, report)
        parts["dev"].extend(chosen[:a.dev])
        parts["test"].extend(chosen[a.dev:])
        if source in public.TRAINABLE:
            train_pool = public.build(source, "train", max(3 * a.train, 800 if not a.smoke else 80), a.seed, revisions[repo])
            parts["train"].extend(select_unique(train_pool, a.train, seen, admission, report))
        manifest["admission"][source] = dict(report)
        print(f"{source}: {dict(report)}", flush=True)

    train_pairs = unique_pairs(a.contrastive_pairs, f"{a.seed}-train", contrastive.TRAINABLE_FAMILIES, seen, admission, "contrastive-train")
    parts["train"] += train_pairs
    eval_pairs = unique_pairs(a.contrastive_eval_pairs, f"{a.seed}-eval", contrastive.TRAINABLE_FAMILIES + contrastive.HELD_OUT_FAMILIES,
                              seen, admission, "contrastive-eval")
    for r in eval_pairs:
        if r["_meta"]["family"] in contrastive.HELD_OUT_FAMILIES:
            r["_meta"]["eval_only"] = True
    for i in range(0, len(eval_pairs), 2):
        parts["dev" if (i // 2) % 2 == 0 else "test"].extend(eval_pairs[i:i + 2])
    print(f"contrastive: {len(train_pairs)} train, {len(eval_pairs)} dev/test", flush=True)

    trees = {f"rand:{composition.canonical(t)}": t for t in composition.sample_trees(a.structures, f"{a.seed}-structures")}
    comp = admit_synthetic(composition.generate(a.groups_per_structure, f"{a.seed}-composition", trees=trees), seen, admission, "composition-train")
    parts["train"] += comp
    comp_dev = admit_synthetic(composition.generate(a.dev_shape_groups, f"{a.seed}-composition-dev"), seen, admission, "composition-dev")
    parts["dev"] += comp_dev
    held = admit_synthetic(composition.generate(a.holdout_shape_groups, f"{a.seed}-composition-holdout",
                                                composition.DEV_SHAPES + composition.TEST_SHAPES, styles=(1, 2),
                                                source="composition_holdout"), seen, admission, "composition-holdout")
    for r in held:
        r["_meta"]["eval_only"] = True
    for i, group in enumerate({r["_meta"]["group_id"]: None for r in held}):
        parts["dev" if i % 2 == 0 else "test"].extend(r for r in held if r["_meta"]["group_id"] == group)
    manifest["composition"] = {"structures": {name: composition.canonical(t) for name, t in trees.items()},
                               "dev_shapes": list(composition.TRAIN_SHAPES),
                               "holdout_shapes": list(composition.DEV_SHAPES + composition.TEST_SHAPES)}
    print(f"composition: {len(comp)} train, {len(comp_dev)} dev, {len(held)} held-out dev/test", flush=True)

    authored = semif.authored_records(Path(a.semif_tasks))
    groups = sorted({r["_meta"]["group_id"] for r in authored})
    random.Random(f"{a.seed}-semif").shuffle(groups)
    n_g = len(groups)
    split_of = {g: ("train" if i < n_g // 2 else "dev" if i < 3 * n_g // 4 else "test") for i, g in enumerate(groups)}
    wanli_eval = semif.wanli_records("test", a.wanli_eval, a.seed)
    wanli_train = semif.wanli_records("train", 2 * a.wanli_train, a.seed)[: a.wanli_train] if a.wanli_train else []
    semif_rows = {"train": [r for r in authored if split_of[r["_meta"]["group_id"]] == "train"] + wanli_train,
                  "dev": [r for r in authored if split_of[r["_meta"]["group_id"]] == "dev"] + wanli_eval[: a.wanli_eval // 2],
                  "test": [r for r in authored if split_of[r["_meta"]["group_id"]] == "test"] + wanli_eval[a.wanli_eval // 2:]}
    for r in authored + wanli_eval:
        r["_meta"]["eval_only"] = False
    for split in ("train", "dev", "test"):
        kept = []
        for r in semif_rows[split]:
            if r["_meta"]["text_sha256"] in seen or not admission.fits(r):
                continue
            seen.add(r["_meta"]["text_sha256"])
            kept.append(r)
        parts[split] += kept
        print(f"semif {split}: {len(kept)} rows " + str(dict(Counter(r['_meta']['source'] for r in kept))), flush=True)
    manifest["trainable_sources"] += ["semif", "wanli"]
    manifest["semif"] = {"authored": len(authored), "authored_groups": n_g, "wanli_eval": len(wanli_eval), "wanli_train": len(wanli_train)}

    for split, tag in (("dev", "dev"), ("test", "test")):
        transfer_rows = []
        if a.mmlu_pro:
            pool = [r for r in transfer.mmlu_pro_records(2 * a.mmlu_pro, a.seed) if r["_meta"]["text_sha256"] not in seen]
            transfer_rows += pool[:a.mmlu_pro]
        if a.buried:
            transfer_rows += transfer.buried_records(parts[split], a.buried, f"{a.seed}-{tag}")
        if a.unknowable_eval_pairs:
            transfer_rows += transfer.unknowable_eval_records(a.unknowable_eval_pairs, f"{a.seed}-{tag}-unknowable")
        kept = []
        for r in transfer_rows:
            if r["_meta"]["text_sha256"] in seen or not admission.fits(r):
                continue
            seen.add(r["_meta"]["text_sha256"])
            kept.append(r)
        parts[split] += kept
        print(f"transfer {split}: {len(kept)} rows " + str(dict(Counter(r['_meta']['source'] for r in kept))), flush=True)

    for split in ("dev", "test"):
        extras, per_source = [], Counter()
        for record in parts[split]:
            source = record["_meta"]["source"]
            if per_source[source] < a.variants_per_source:
                variants = contrast_cases(record, a.seed)
                for v in variants:
                    if not admission.fits(v):
                        raise ValueError(f"variant of {record['_meta']['id']} exceeds the context")
                extras.extend(variants)
                per_source[source] += bool(variants)
        parts[split].extend(extras)

    if not a.no_night2:
        eval_hashes = {r["_meta"]["text_sha256"] for s in ("dev", "test") for r in parts[s]}
        night = (night2.dates_records(a.night2_date_pairs, f"{a.seed}-night2")
                 + night2.unknowable_records(a.night2_unknowable_pairs, f"{a.seed}-night2")
                 + night2.assertion_records(parts["train"], a.night2_assertions, f"{a.seed}-night2"))
        kept = [r for r in night if r["_meta"]["text_sha256"] not in eval_hashes and semantic_hash(r) not in eval_hashes]
        for r in kept:
            if not admission.fits(r):
                raise ValueError(f"night2 record {r['_meta']['id']} exceeds the context")
        parts["train"] += kept
        manifest["night2"] = {"records": len(kept), "dropped_eval_overlap": len(night) - len(kept),
                              "sources": sorted({r["_meta"]["source"] for r in kept})}
        print(f"night2: {len(kept)} train records ({len(night) - len(kept)} dropped for eval overlap)", flush=True)

    fractions = {k: a.rl_fraction for k in RL_FRACTION} if a.rl_fraction is not None else RL_FRACTION
    parts["train"], parts["rl"] = reserve_rl(parts["train"], a.seed, fractions, a.rl_target if not a.smoke else 200)
    manifest["rl_fraction"] = fractions
    manifest["rl_target"] = a.rl_target
    manifest["sft_budget"] = SFT_BUDGET
    for split in ("train", "rl"):
        random.Random(f"{a.seed}-{split}").shuffle(parts[split])
    for split, records in parts.items():
        validate(records)
        path = out / f"{split}.jsonl"
        write_jsonl(path, records)
        manifest["files"][path.name] = {
            "records": len(records), "questions": sum(len(r["questions"]) for r in records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sources": dict(sorted(Counter(r["_meta"]["source"] for r in records).items())),
            "question_sources": dict(sorted(Counter(q["src"] for r in records for q in r["questions"].values()).items())),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"records": v["records"], "questions": v["questions"]} for k, v in manifest["files"].items()}, indent=2))


if __name__ == "__main__":
    main()
