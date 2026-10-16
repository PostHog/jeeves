"""Build Jeeves' training data from public sources: `python -m prep.prep` writes train/dev/rl/test.jsonl and manifest.json.

Every source is converted into a pool of rows keyed by id: v1's data (the public datasets in public.py and the synthetic
generators), third-party decision sets (thirdparty.py), decider's converted datasets (decider.py), general-skill datasets
(skills.py) and prompt-injection data (injection.py). train/dev/rl.jsonl are then read off the ordered id lists in
manifests/<mix>/ and checked against the sha256 hashes there. Rows removed during decontamination are simply absent
from the id lists.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prep import composition, contrastive, decider, injection, mix, night2, public, semif, skills, thirdparty, transfer
from prep.format import DataFormat, option_text, render, state_hash, write_jsonl

SPLITS = ("train", "rl", "dev", "test")
MANIFESTS = Path(__file__).resolve().parent / "manifests"
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
        if variant == "none_absent":
            rq["criteria"].pop(rq["label"])
            rq["label"] = NONE_KEY
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


# v1's data, which v2's mix includes as is
V1 = argparse.Namespace(
    seed="20260918", train=4500, dev=80, test=80, variants_per_source=12, contrastive_pairs=448, contrastive_eval_pairs=16,
    structures=60, groups_per_structure=28, dev_shape_groups=4, holdout_shape_groups=8, night2_date_pairs=600,
    night2_unknowable_pairs=60, night2_assertions=3200, no_night2=False, mmlu_pro=200, buried=80, unknowable_eval_pairs=5,
    wanli_eval=256, wanli_train=4500, rl_target=10000, sources=",".join(public.TRAINABLE + public.EVAL_ONLY),
    tokenizer="Qwen/Qwen3.5-9B")
SEMIF = ("https://raw.githubusercontent.com/TheoLeeCJ/SemIf-OpenJev/23cf1f39fc9534fe81437200959b6dfc7106e45a/benchmarks/data/authored144.jsonl",
         "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079")


def semif_tasks(dl: Path) -> Path:
    path = dl / "semif" / "authored144.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SEMIF[0], path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SEMIF[1]:
        raise ValueError(f"{path} does not match the pinned SemIf tasks")
    return path


def build_v1(a: argparse.Namespace, semif_tasks: Path) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    sources = [s for s in a.sources.split(",") if s]
    admission = Admission(a.tokenizer)
    revisions = resolve_revisions(sources)
    parts: dict[str, list[dict]] = {s: [] for s in SPLITS}
    seen: set[str] = set()
    manifest: dict[str, Any] = {"seed": a.seed, "tokenizer": a.tokenizer,
                                "context": {"max_state": MAX_STATE, "max_branch": MAX_BRANCH, "max_packed": MAX_PACKED,
                                            "admission_headroom": HEADROOM},
                                "trainable_sources": [s for s in sources if s in public.TRAINABLE],
                                "eval_only_sources": [s for s in sources if s in public.EVAL_ONLY],
                                "dataset_revisions": revisions, "admission": {}}

    for source in sources:
        report: Counter = Counter()
        repo = public.REPOS[source].partition(":")[0]
        eval_pool = public.build(source, "eval", max(3 * (a.dev + a.test), 600), a.seed, revisions[repo])
        chosen = select_unique(eval_pool, a.dev + a.test, seen, admission, report)
        parts["dev"].extend(chosen[:a.dev])
        parts["test"].extend(chosen[a.dev:])
        if source in public.TRAINABLE:
            train_pool = public.build(source, "train", max(3 * a.train, 800), a.seed, revisions[repo])
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

    authored = semif.authored_records(semif_tasks)
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

    parts["train"], parts["rl"] = reserve_rl(parts["train"], a.seed, RL_FRACTION, a.rl_target)
    manifest["rl_fraction"] = RL_FRACTION
    manifest["rl_target"] = a.rl_target
    manifest["sft_budget"] = SFT_BUDGET
    for split in ("train", "rl"):
        random.Random(f"{a.seed}-{split}").shuffle(parts[split])
    for records in parts.values():
        validate(records)
    return parts, manifest


def build_pool(dl: Path, v1: dict[str, list[dict]], pi_ids: list[str]) -> mix.Pool:
    pool = mix.Pool()
    for split in ("train", "rl", "dev", "test"):
        pool.add(v1[split])
    for rows in thirdparty.build(dl / "thirdparty").values():
        pool.add(rows)
    pool.add(decider.build(dl / "decider", [mix.as_read(r) for r in v1["dev"] + v1["test"]]))
    for train, dev in skills.build(dl / "skills").values():
        pool.add(train + dev, ascii=True)
    pool.add(injection.build(dl / "injection", pi_ids, list(pool.rows.values())), ascii=True)
    return pool


def injection_ids(ids: list[str]) -> list[str]:
    """The injection set's ids in row order: its rows carry their row number."""
    return sorted({i for i in ids if i.startswith("pi_")}, key=lambda i: int(i.split("/")[1].split("_")[0]))


def use_hf_cache(root: Path) -> None:
    """Keep Hugging Face downloads under the download directory unless HF_HOME or HF_HUB_CACHE is set."""
    if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
        return
    import datasets.config
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_CACHE = str(root / "hub")
    huggingface_hub.constants.HF_XET_CACHE = str(root / "xet")
    datasets.config.HF_DATASETS_CACHE = root / "datasets"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download the public datasets at pinned revisions and build Jeeves' training data.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--downloads", help="download directory (default: <out>/downloads)")
    ap.add_argument("--ids", default=str(MANIFESTS / "v2"), help="directory of the mix's id lists and expected hashes")
    ap.add_argument("--make-ids", metavar="MIX", help="write --ids from the train/dev/rl.jsonl files of an existing mix instead")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out, ids_dir = Path(a.out), Path(a.ids)
    dl = Path(a.downloads) if a.downloads else out / "downloads"
    use_hf_cache(dl / "hf")
    if out.exists() and any(p.suffix == ".jsonl" for p in out.iterdir()) and not a.force:
        raise FileExistsError(f"{out} already has data; pass --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    if a.make_ids:
        ids = {s: [json.loads(line)["_meta"]["id"] for line in open(Path(a.make_ids) / f"{s}.jsonl")] for s in mix.SPLITS}
        expected = {}
    else:
        ids = {s: mix.read_ids(ids_dir / f"{s}.ids.gz") for s in mix.SPLITS}
        expected = json.loads((ids_dir / "manifest.json").read_text())["files"]

    v1, v1_manifest = build_v1(V1, semif_tasks(dl))
    pool = build_pool(dl, v1, injection_ids([i for s in mix.SPLITS for i in ids[s]]))

    if a.make_ids:
        for s in mix.SPLITS:
            ids[s], unmatched = mix.ids_of(Path(a.make_ids) / f"{s}.jsonl", pool)
            if unmatched:
                raise ValueError(f"{s}: {len(unmatched)} rows of {a.make_ids} are not in the pool, e.g. "
                                 f"{[r['_meta']['id'] for r in unmatched[:5]]}")
            mix.write_ids(ids_dir / f"{s}.ids.gz", ids[s])
            expected[f"{s}.jsonl"] = {"records": len(ids[s]),
                                      "sha256": hashlib.sha256((Path(a.make_ids) / f"{s}.jsonl").read_bytes()).hexdigest()}

    files = {f"{s}.jsonl": mix.write_split(pool, ids[s], out / f"{s}.jsonl") for s in mix.SPLITS}
    write_jsonl(out / "test.jsonl", v1["test"])
    files["test.jsonl"] = {"records": len(v1["test"]), "sha256": hashlib.sha256((out / "test.jsonl").read_bytes()).hexdigest()}
    if a.make_ids:
        expected["test.jsonl"] = {k: files["test.jsonl"][k] for k in ("records", "sha256")}
        (ids_dir / "manifest.json").write_text(json.dumps({"files": expected}, indent=2) + "\n")
    sources = {"thirdparty": {"openjev_v1.1": thirdparty.OJ11, "openjev": thirdparty.OJ, "kev": thirdparty.KEV,
                              "kev_github": thirdparty.KEV_GITHUB, "spark": thirdparty.SPARK, "typed_decisions": thirdparty.TYPED},
               "decider": {"code": [decider.REPO, decider.COMMIT], "datasets": decider.PROVENANCE},
               "skills": {n: v[:3] for n, v in skills.SOURCES.items()},
               "injection": {n: v[:2] for n, v in injection.REPOS.items()},
               "semif_tasks": SEMIF}
    manifest = {"ids": ids_dir.name, "files": files, "pool_rows": len(pool.rows), "sources": sources, "v1": v1_manifest}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    bad = [f for f, e in expected.items() if files[f]["sha256"] != e["sha256"]]
    print(json.dumps({f: {k: v[k] for k in ("records", "sha256")} for f, v in files.items()}, indent=2))
    if bad:
        raise ValueError(f"{bad} do not match the expected sha256 in {ids_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
