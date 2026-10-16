"""Third-party decision datasets that are already close to Jeeves' format: Open-Jev, kev, decider's teacher data, spark and
typed-decisions. Every source is pinned to a revision; (state, instruction) pairs are deduplicated across all of them."""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from prep import decider
from prep.format import Question, render, state_hash

MAX_STATE = 24000
OJ11 = ("ZefanCai/Open-Jev-v1.1", "10ad6888333fa97f8c948192797bad3de3040802")
OJ = ("ZefanCai/Open-Jev", "c67699e13d0ae25e35b77165a4b6b079bedc8aba")
KEV = ("jaredpalmer/kev-suites", "cc4bac803e73112689ec327ffa481c519cbc7a05")
KEV_GITHUB = ("https://github.com/jaredpalmer/kev", "fc4a17e194bca35a48c57211392ec49ea54a94ac")
SPARK = ("abhishek085/spark-s1-osdg-v1", "11b3bc74545c602c7fb1b2c35a9c28d12031798d")
TYPED = ("LocalLLaMA/typed-decisions", "e039ebffcc280174dd354227424fb2b249f191de")
TYPESAFE = re.compile(r"typesafe", re.I)
OJ_EXCLUDED = ["wanli", "customer-control", "wikispeedia", "wikiracing"]
BUCKET = re.compile(r"(support|help|question|account|service|assist|feedback|inquir|enquir|contact|issue|request|info|general|misc|"
                    r"other|else|query|queries|concern|complaint|problem|advice|guidance|topics?|customer|care|desk|admin|office|"
                    r"reception)", re.I)


def download(dl: Path) -> dict[str, Path]:
    get = lambda repo, rev, patterns, name: Path(snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=patterns,
                                                                   local_dir=dl / name))
    paths = {"openjev11": get(*OJ11, ["data/community-hard-mix-v2-redistributable/*.parquet"], "openjev11"),
             "openjev": get(*OJ, ["data/*/*.parquet"], "openjev"),
             "kev": get(*KEV, ["hard-v1/train.jsonl", "documents-v1/train.jsonl"], "kev"),
             "spark": get(*SPARK, ["jev/train.jsonl", "jev/train_option_augmented.jsonl"], "spark"),
             "typed": get(*TYPED, ["all/*.parquet"], "typed"),
             "decider": decider.fetch(dl / "decider")}
    kevgh = dl / "kev_github" / "devtools-v1-train.jsonl"
    if not kevgh.exists():
        kevgh.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_GITHUB[1]}/evals/devtools-v1/train.jsonl"
        urllib.request.urlretrieve(url, kevgh)
    paths["kev_devtools"] = kevgh
    return paths


def check_q(q, drop):
    """Validated copy of a question dict, or None."""
    t = q.get("type")
    try:
        if t == "choice":
            c = q["criteria"]
            if not isinstance(c, dict) or len(c) < 2:
                drop["choice<2opts"] += 1; return None
            if len(c) > 255:
                drop["choice>255opts"] += 1; return None
            if q["label"] not in c:
                drop["bad_label"] += 1; return None
        elif t == "noul":
            c = q.get("criteria")
            if c is not None and (not isinstance(c, dict) or not set(c) <= {"true", "false"}):
                drop["bad_noul_criteria"] += 1; return None
            lab = q["label"]
            if isinstance(lab, bool):
                lab = int(lab)
            if isinstance(lab, str) and lab.lower() in ("true", "yes", "1", "false", "no", "0"):
                lab = int(lab.lower() in ("true", "yes", "1"))
            if lab not in (0, 1):
                drop["bad_label"] += 1; return None
            q["label"] = lab
        elif t == "score":
            c = q["criteria"]
            if not isinstance(c, list) or len(c) < 2:
                drop["score<2levels"] += 1; return None
            lab = int(q["label"])
            if not 0 <= lab < len(c):
                drop["bad_label"] += 1; return None
            q["label"] = lab
        else:
            drop["bad_type"] += 1; return None
        if q.get("instructions") in (None, ""):
            drop["no_instructions"] += 1; return None
        qq = Question(id="x", type=t, instructions=q["instructions"], criteria=q.get("criteria"), label=q["label"],
                      src=q.get("src", ""), target=q.get("target"))
        qq.label_index(); qq.options()
        if q.get("target") is not None:
            if set(q["target"]) - set(qq.keys()):
                drop["bad_target_keys"] += 1; return None
            qq.target_vector()
    except (KeyError, ValueError, TypeError) as e:
        drop["malformed:" + type(e).__name__] += 1
        return None
    out = {"type": t, "instructions": q["instructions"]}
    if q.get("criteria") is not None or t != "noul":
        out["criteria"] = q.get("criteria")
    out["label"] = q["label"]
    out["src"] = q.get("src", "")
    if q.get("target") is not None:
        out["target"] = q["target"]
    return out


class Builder:
    def __init__(self):
        self.seen: set = set()  # (text_sha256, instruction text) across all sets
        self.sets: dict[str, list[dict]] = {}

    def emit(self, name, records, *, repo, revision, licence, drop):
        """records: dicts of state, questions, meta. Validates, drops duplicates and stores the set."""
        out, ids = [], set()
        for r in records:
            text = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], sort_keys=True, ensure_ascii=False)
            if len(text) > MAX_STATE:
                drop["state>24000"] += len(r["questions"]); continue
            sha = state_hash(r["state"])
            keep = {}
            for qid, q in r["questions"].items():
                q = check_q(q, drop)
                if q is None:
                    continue
                key = (sha, render(q["instructions"]))
                if key in self.seen:
                    drop["duplicate"] += 1; continue
                self.seen.add(key)
                keep[qid] = q
            if not keep:
                continue
            meta = dict(r["meta"])
            if meta["id"] in ids:
                base, n = meta["id"], 2
                while f"{base}#{n}" in ids:
                    n += 1
                meta["id"] = f"{base}#{n}"
            ids.add(meta["id"])
            meta.setdefault("repo", repo); meta.setdefault("revision", revision); meta.setdefault("licence", licence)
            meta.update(text_sha256=sha, split="train")
            out.append({"state": r["state"], "questions": keep, "_meta": meta})
        self.sets[name] = out
        print(f"{name}: {len(out)} rows, dropped {dict(drop)}", flush=True)


# ---------------------------------------------------------------- Open-Jev
def oj_question(rec, src_tag):
    kind, opts, tgt = rec["kind"], list(rec["options"]), [float(x) for x in rec["target"]]
    if len(opts) != len(tgt) or sum(tgt) <= 0:
        return None
    best = max(range(len(tgt)), key=lambda i: tgt[i])
    soft = max(tgt) < 0.999999
    q = {"instructions": rec["question"], "src": src_tag}
    if kind == "noul":
        lo = [o.lower() for o in opts]
        if lo == ["no", "yes"]:
            pf, pt = tgt
        elif lo == ["yes", "no"]:
            pt, pf = tgt
        else:
            return None
        q.update(type="noul", criteria=None, label=int(pt > pf))
        if soft:
            q["target"] = {"false": pf, "true": pt}
    elif kind == "choice":
        if len(set(opts)) != len(opts):
            return None
        q.update(type="choice", criteria={o: None for o in opts}, label=opts[best])
        if soft:
            q["target"] = {o: p for o, p in zip(opts, tgt)}
    elif kind == "score":
        q.update(type="score", criteria=opts, label=best)
        if soft:
            q["target"] = {str(i): p for i, p in enumerate(tgt)}
    else:
        return None
    return q


def oj_rows(parquet, drop):
    for r in pq.read_table(parquet, columns=["record_json"]).to_pylist():
        rec = json.loads(r["record_json"])
        if rec.get("split") != "train":
            drop["not_train"] += 1; continue
        src = rec["source"]
        if any(src.startswith(e) for e in OJ_EXCLUDED):
            drop["excluded_source:" + src] += 1; continue
        if TYPESAFE.search(json.dumps([rec["state"], rec["question"], rec["options"]], ensure_ascii=False)):
            drop["typesafe_text"] += 1; continue
        q = oj_question(rec, "openjev/" + src)
        if q is None:
            drop["malformed_source_row"] += 1; continue
        lic = ((rec.get("metadata") or {}).get("provenance") or {}).get("license", "CC0-1.0")
        yield rec, q, lic


def oj_merge(items, config, repo, revision):
    """Rows sharing an identical state (within a config) become one multi-question record."""
    groups = {}
    for rec, q, lic in items:
        sha = state_hash(rec["state"])
        g = groups.get(sha)
        if g is None:
            g = groups[sha] = {"state": rec["state"], "questions": {}, "meta": {
                "source": "openjev/" + config, "id": f"{repo}:{config}:{rec['id']}", "group_id": rec.get("group_id"),
                "orig_source": rec["source"], "licence": lic}}
        base = str((rec.get("metadata") or {}).get("question_id") or rec["kind"])
        qid, n = base, 1
        while qid in g["questions"]:
            n += 1; qid = f"{base}#{n}"
        g["questions"][qid] = q
    for g in groups.values():
        g["meta"].update(repo=repo, revision=revision)
    return list(groups.values())


def openjev(b: Builder, p11: Path, p: Path):
    drop = collections.Counter()
    recs = oj_merge(list(oj_rows(glob.glob(f"{p11}/data/community-hard-mix-v2-redistributable/train-*.parquet")[0], drop)),
                    "v1.1/community-hard-mix-v2-redistributable", *OJ11)
    # the older release: every config except release-v2 (a strict subset of browser-drone-expansion)
    for cfg in sorted(x.name for x in (p / "data").iterdir()):
        if cfg == "release-v2-redistributable":
            continue
        cand = []
        seen = {(state_hash(r["state"]), render(q["instructions"])) for r in recs for q in r["questions"].values()}
        for rec, q, lic in oj_rows(glob.glob(f"{p}/data/{cfg}/train-*.parquet")[0], drop):
            key = (state_hash(rec["state"]), render(q["instructions"]))
            if key in seen:
                drop["dup_of_v1.1_or_earlier_config"] += 1; continue
            seen.add(key)
            cand.append((rec, q, lic))
        # hash-ordered round robin across sub-sources, capped at 30k questions per config
        by = collections.defaultdict(list)
        for it in cand:
            by[it[0]["source"]].append(it)
        for v in by.values():
            v.sort(key=lambda it: hashlib.sha256(it[0]["id"].encode()).hexdigest())
        chosen, i = [], 0
        while len(chosen) < 30000 and any(len(v) > i for v in by.values()):
            for s in sorted(by):
                if len(by[s]) > i and len(chosen) < 30000:
                    chosen.append(by[s][i])
            i += 1
        recs = recs + oj_merge(chosen, "v1/" + cfg, *OJ)
    b.emit("openjev", recs, repo=f"{OJ11[0]} + {OJ[0]}", revision=f"Open-Jev-v1.1@{OJ11[1]}; Open-Jev@{OJ[1]}",
           licence="CC0-1.0", drop=drop)


# ---------------------------------------------------------------- kev
def kev_records(path, drop, prefix, keep_sources=None):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        m = d.get("_meta", {})
        if m.get("split", "train") != "train":
            drop["not_train"] += 1; continue
        if keep_sources is not None and m.get("source") not in keep_sources:
            drop["excluded_source:" + str(m.get("source"))] += len(d["questions"]); continue
        qs = {}
        for qid, q in d["questions"].items():
            q = dict(q)
            q.setdefault("src", m.get("source", ""))
            q["src"] = f"{prefix}/{q['src']}"
            qs[qid] = q
        meta = {"source": prefix, "id": f"{prefix}:{m.get('id')}"}
        for k in ("group_id", "family", "variant", "template", "subtype", "project", "repo_licence", "length_bucket"):
            if k in m:
                meta[k] = m[k]
        meta["orig_source"] = m.get("source")
        yield {"state": d["state"], "questions": qs, "meta": meta}


def kev(b: Builder, p: Path, devtools: Path):
    drop = collections.Counter()
    b.emit("kev_hard", kev_records(p / "hard-v1/train.jsonl", drop, "kev/hard-v1"), repo=KEV[0], revision=KEV[1],
           licence="Apache-2.0 (github.com/jaredpalmer/kev repo licence; HF dataset card declares none)", drop=drop)
    drop = collections.Counter()
    b.emit("kev_documents", kev_records(p / "documents-v1/train.jsonl", drop, "kev/documents-v1"), repo=KEV[0], revision=KEV[1],
           licence="Public domain text (US CFPB complaints via davidheineman/consumer-finance-complaints-large); suite Apache-2.0",
           drop=drop)
    # aegis is dropped: its card states a research-purpose intent
    drop = collections.Counter()
    b.emit("kev_devtools", kev_records(devtools, drop, "kev/devtools-v1", keep_sources={"codereviewer", "commitpackft", "flakeflagger"}),
           repo=f"{KEV_GITHUB[0]} (evals/devtools-v1/train.jsonl)", revision=KEV_GITHUB[1],
           licence="codereviewer CC-BY-4.0 (permissive-licence repos only); commitpackft MIT + per-row permissive repo licence; "
                   "flakeflagger CC-BY-4.0/BSD-3 (non-permissive projects dropped upstream)", drop=drop)


# ---------------------------------------------------------------- decider teacher data
def decider_teacher(b: Builder, root: Path):
    """decider's teacher-labelled sets, with the agreement filters of decider's own load_teacher() (decider/data/mixture.py)."""
    drop = collections.Counter()
    load = lambda f: [json.loads(l) for l in open(root / "teacher_data" / f, encoding="utf-8") if l.strip()]

    def conv(r, name, i):
        qs = {}
        for k, q in enumerate(r["questions"]):
            crit = q.get("criteria")
            if q["type"] == "choice" and isinstance(crit, dict):
                crit = {kk: (None if v in ("null", "", None) else v) for kk, v in crit.items()}
            qq = {"type": q["type"], "instructions": q["instructions"], "label": q["answer"], "src": f"decider/{name}"}
            if crit is not None or q["type"] != "noul":
                qq["criteria"] = crit
            qs[f"q{k}"] = qq
        meta = {"source": f"decider/{name}", "id": f"decider/{name}:{i}"}
        meta.update({kk: r[kk] for kk in ("domain", "kind", "recipe", "group") if r.get(kk) is not None})
        return {"state": r["state"], "questions": qs, "meta": meta}

    def agreed(r):
        q = r["questions"][0]
        return q["teacher_ok"] or (r["group"] == "generic" and q["teacher_pred"] == r["catchall"])

    out = [conv(r, "custom_questions", i) for i, r in enumerate(load("custom_questions.jsonl"))]
    for i, r in enumerate(load("routing_messages.jsonl")):
        if agreed(r):
            out.append(conv(r, "routing_messages", i))
    terse = load("routing_terse.jsonl")
    generic = {r["questions"][0]["instructions"] + json.dumps(list(r["questions"][0]["criteria"])): r["questions"][0]["answer"]
               for r in terse if r["group"] == "generic"}
    for i, r in enumerate(terse):
        q = r["questions"][0]
        if BUCKET.search(generic.get(q["instructions"] + json.dumps(list(q["criteria"])), "") or "") and agreed(r):
            out.append(conv(r, "routing_terse", i))
    order = ["safe", "caution", "destructive"]
    for i, r in enumerate(load("commands.jsonl")):
        q = r["questions"][0]; pred = q.get("teacher_pred")
        if q["teacher_ok"] or (pred in order and abs(order.index(pred) - order.index(q["answer"])) == 1):
            out.append(conv(r, "commands", i))
    for i, j in enumerate(load("situations.jsonl")):  # as in decider's tasks_agents._synth
        opts = [str(o) for o in j["options"]]
        if len(set(opts)) != len(opts) or not 0 <= int(j["answer"]) < len(opts):
            continue
        qs = {"action": {"type": "choice", "instructions": j["question"], "criteria": {o: None for o in opts},
                         "label": opts[int(j["answer"])], "src": "decider/situations"}}
        if "danger" in j:
            qs["danger"] = {"type": "noul", "instructions": "Is the agent in immediate danger?", "label": int(bool(j["danger"])),
                            "src": "decider/situations_danger"}
        out.append({"state": f"Setting: {j.get('domain', '')}.\n{j['situation']}", "questions": qs,
                    "meta": {"source": "decider/situations", "id": f"decider/situations:{i}", "domain": j.get("domain")}})
    b.emit("decider_teacher", out, repo=f"{decider.REPO} (teacher_data/)",
           revision=decider.COMMIT, licence="Apache-2.0 (repo); labels/text from local Qwen3.5-27B teacher (Apache-2.0 weights)",
           drop=drop)


# ---------------------------------------------------------------- spark
def spark_row(r, fname, i, drop):
    qd, tg = r["question"], r["target"]
    t, prompt, src = qd["type"], qd["prompt"], f"spark/{r['domain']}"
    dist = tg.get("dist") or {}
    soft = dist and max(dist.values()) < 0.999999
    if t == "choice":
        opts = list(qd["options"])
        defs = {}
        if "\nOption definitions:\n" in prompt:
            prompt, block = prompt.split("\nOption definitions:\n", 1)
            for ln in block.splitlines():
                m = re.match(r"^- (.+?): (.*)$", ln)
                if m:
                    defs[m.group(1)] = m.group(2)
        if defs and set(defs) != set(opts):
            drop["spark_defs_mismatch"] += 1; return None
        q = {"type": "choice", "instructions": prompt, "criteria": {o: defs.get(o) for o in opts}, "label": tg["label"], "src": src}
        if soft:
            q["target"] = {o: float(dist.get(o, 0)) for o in opts}
    elif t == "noul":
        q = {"type": "noul", "instructions": prompt, "label": int(str(tg["label"]).lower() in ("yes", "true", "1")), "src": src}
        if soft:
            q["target"] = {"false": float(dist.get("no", 0)), "true": float(dist.get("yes", 0))}
    elif t == "score":
        levels = list(qd["levels"])
        rub = {}
        for ln in (qd.get("rubric") or "").splitlines():
            m = re.match(r"^(\S+?): (.*)$", ln)
            if m:
                rub[m.group(1)] = m.group(2)
        q = {"type": "score", "instructions": prompt, "criteria": [rub.get(l, l) for l in levels],
             "label": levels.index(str(tg["label"])), "src": src}
        if soft:
            q["target"] = {str(k): float(dist.get(l, 0)) for k, l in enumerate(levels)}
    else:
        drop["bad_type"] += 1; return None
    state = r["state"]["content"] if isinstance(r["state"], dict) and "content" in r["state"] else r["state"]
    return {"state": state, "questions": {"decision": q},
            "meta": {"source": "spark", "id": f"spark/{fname}:{r['id']}:{i}", "task_pack": r["meta"].get("task_pack"),
                     "label_quality": r["meta"].get("label_quality"), "scenario_family": r["meta"].get("scenario_family")}}


def spark(b: Builder, p: Path):
    """The option-augmented file only re-orders the options of the same rows, so the dedup removes all of it."""
    drop, out = collections.Counter(), []
    for fname in ("train.jsonl", "train_option_augmented.jsonl"):
        for i, l in enumerate(open(p / "jev" / fname, encoding="utf-8")):
            if l.strip():
                r = json.loads(l)
                if r["meta"].get("split", "train") != "train":
                    drop["not_train"] += 1; continue
                c = spark_row(r, fname, i, drop)
                if c:
                    out.append(c)
    b.emit("spark", out, repo=SPARK[0], revision=SPARK[1], licence="Apache-2.0", drop=drop)


# ---------------------------------------------------------------- typed-decisions
def typed(b: Builder, p: Path):
    """Soft targets are the mean of three teacher samples (gold.probabilities), labels gold.label."""
    drop, out = collections.Counter(), []
    for r in pq.read_table(glob.glob(f"{p}/all/train-*.parquet")[0]).to_pylist():
        if r["split"] != "train":
            drop["not_train"] += 1; continue
        try:
            state = json.loads(r["state"])
        except ValueError:
            state = r["state"]
        qs_in, gold = json.loads(r["questions"]), json.loads(r["gold"])
        qs = {}
        for qid, q in qs_in.items():
            g = gold.get(qid)
            if g is None:
                drop["no_gold"] += 1; continue
            p_ = g.get("probabilities") or {}
            qq = {"type": q["type"], "instructions": q["instructions"], "src": f"typed_decisions/{r['workflow']}/{qid}"}
            if q.get("criteria") is not None or q["type"] != "noul":
                qq["criteria"] = q.get("criteria")
            if q["type"] == "choice":
                qq["label"] = g["label"]
                qq["target"] = {k: float(p_.get(k, 0)) for k in q["criteria"]}
            elif q["type"] == "noul":
                qq["label"] = int(str(g["label"]).lower() in ("true", "yes", "1"))
                qq["target"] = {"false": float(p_.get("false", 0)), "true": float(p_.get("true", 0))}
            else:
                qq["label"] = int(g["label"])
                qq["target"] = {str(i): float(p_.get(str(i), 0)) for i in range(len(q["criteria"]))}
            qs[qid] = qq
        out.append({"state": state, "questions": qs,
                    "meta": {"source": "typed_decisions", "id": f"typed_decisions:{r['id']}", "workflow": r["workflow"]}})
    b.emit("typed_decisions", out, repo=TYPED[0], revision=TYPED[1], licence="Apache-2.0", drop=drop)


def build(dl: Path) -> dict[str, list[dict]]:
    paths = download(dl)
    b = Builder()
    openjev(b, paths["openjev11"], paths["openjev"])
    kev(b, paths["kev"], paths["kev_devtools"])
    decider_teacher(b, paths["decider"])
    spark(b, paths["spark"])
    typed(b, paths["typed"])
    return b.sets
