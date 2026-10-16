"""decider_public: public decision datasets converted by the decider project's own task loaders and augmentations.

The code comes from github.com/Mapika/decider (Apache-2.0) at COMMIT: it is downloaded, imported and run unchanged, with
every dataset load pinned to the revision recorded when the set was first built. Stage 1 runs decider's converters
(train splits only), stage 2 caps each task's pool, applies decider's augmentations and converts to Jeeves records.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

import datasets

COMMIT = "e50e549b47e2da69223734fee4efa1ddd4528e93"
REPO = "https://github.com/Mapika/decider"
REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453", "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "LabHC/bias_in_bios": "052f01de644dba841176e0449528b41f27d94a61", "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "google/civil_comments": "f2970eb3a55777454c94069077cc8d9b5866312d", "clinc/clinc_oos": "155b9c710419136e17307b80d0a13e68cd46b4ec",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155", "fancyzhx/dbpedia_14": "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
    "google-research-datasets/go_emotions": "add492243ff905527e67aeb8b80c082af02207c3",
    "ucberkeley-dlab/measuring-hate-speech": "5468f6e118396646b02a2f691e771f6b6d9502ea",
    "nvidia/HelpSteer2": "990b2711a36180dd19d9c94b8627844866f8982a", "nvidia/HelpSteer3": "f6d145777bcbde96137596340fab89793acd1031",
    "Anthropic/hh-rlhf": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa", "chengxuphd/liar2": "12ad1a68e4256d0fab1701396e18873393cdc248",
    "openlifescienceai/medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    "GBaker/MedQA-USMLE-4-options": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
    "osunlp/Mind2Web": "17ece8eb89862368edc0cc806acee6fca5163474", "allenai/prosocial-dialog": "d77d7ad3c624c51030f2f32c83e892b3d620b3d4",
    "allenai/qasc": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "stanfordnlp/snli": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
}
MASSIVE = ("AmazonScience/massive", "ed58ac423a2f4121720918bf5301577edce4ffd3")  # refs/convert/parquet commit
LICENCE = {  # HF card licence at the pinned revision
    "clinc_oos": "cc-by-3.0", "banking77": "mit", "massive_intent_upstream": "cc-by-4.0", "dbpedia": "cc-by-sa-3.0",
    "go_emotions": "apache-2.0", "civil_comments": "cc0-1.0", "snli": "cc-by-sa-4.0", "boolq": "cc-by-sa-3.0",
    "arc": "cc-by-sa-4.0", "commonsense_qa": "mit", "qasc": "cc-by-4.0", "medqa": "cc-by-4.0", "bias_in_bios": "mit",
    "helpsteer2": "cc-by-4.0", "helpsteer3_pref": "cc-by-4.0", "hate_speech_scales": "cc-by-4.0", "liar2": "apache-2.0",
    "prosocial_safety": "cc-by-4.0", "hh_rlhf": "mit", "medmcqa": "apache-2.0", "mind2web": "cc-by-4.0",
}
TASK_REPOS = {
    "arc": "allenai/ai2_arc", "banking77": "mteb/banking77", "bias_in_bios": "LabHC/bias_in_bios", "boolq": "google/boolq",
    "civil_comments": "google/civil_comments", "clinc_oos": "clinc/clinc_oos", "commonsense_qa": "tau/commonsense_qa",
    "dbpedia": "fancyzhx/dbpedia_14", "go_emotions": "google-research-datasets/go_emotions",
    "hate_speech_scales": "ucberkeley-dlab/measuring-hate-speech", "helpsteer2": "nvidia/HelpSteer2", "helpsteer3_pref": "nvidia/HelpSteer3",
    "hh_rlhf": "Anthropic/hh-rlhf", "liar2": "chengxuphd/liar2", "massive_intent_upstream": MASSIVE[0],
    "medmcqa": "openlifescienceai/medmcqa", "medqa": "GBaker/MedQA-USMLE-4-options", "mind2web": "osunlp/Mind2Web",
    "prosocial_safety": "allenai/prosocial-dialog", "qasc": "allenai/qasc", "snli": "stanfordnlp/snli",
}
PROVENANCE = {t: (repo, MASSIVE[1] if repo == MASSIVE[0] else REVISIONS[repo], LICENCE[t]) for t, repo in TASK_REPOS.items()}
# Raw rows (sha256 of the normalised text) that the original build removed during decontamination.
# Capping happens after this filter, so without them the pools, and every random draw after them, would differ.
EXCLUDED_ROWS = {
    "civil_comments": {
        "07123e1f482356c415f684407a3b8723e10b2cbbc0b8fcd6282c49d37c9c1abc",
        "808cffc825ab518424fe545b2c361f022cee47d50d633718c98e87790b8810b1",
        "8214f3d09939c0691b95f0f16e0386a5395daa3359a637062e233149f82bf5a2",
        "8e19be0d75ff14c0dc6b2691ddfafcfc2e083e8d276563444df6f5e0230e1dfc",
    },
    "helpsteer3_pref": {
        "87f7b30102167fd8807d14e9300e3f196571ee690c0f4df5fb2f75952194da01",
    },
    "hh_rlhf": {
        "2edcde2d86f11491c45f8e327868a2ab0d7dc4295fe902dff19e1a09104050eb",
        "153cf98181838977980a8e0883030e07d71f16f9aa4aab4613593b791b7c9440",
    },
}

SEED, CAP, MAX_STATE, AUG_MAX_FRAC = 20261009, 8000, 24000, 0.25
LETTER_TASKS = {"arc", "commonsense_qa", "qasc", "medqa", "medmcqa", "mind2web"}
# augmentation row budgets (rows, before dedupe); total kept <= AUG_MAX_FRAC of the output
BUDGET = dict(wide=5000, padded=4000, described=8000, single=3000, abstain=3000, json=4000, json_indexed=2500, isolated=4000, rules=5000)
BIG_LABEL = {"clinc_oos", "banking77", "massive_intent", "dbpedia", "go_emotions", "bias_in_bios"}
SCALE_TASKS = {"helpsteer2", "helpsteer3_pref", "hate_speech_scales", "liar2", "prosocial_safety"}
MODES = ["named", "opaque", "named_json", "opaque_json"]
NG = 13


def fetch(dl: Path) -> Path:
    """Download decider's source at COMMIT; returns the source directory."""
    root = dl / f"decider-{COMMIT}"
    if not root.exists():
        dl.mkdir(parents=True, exist_ok=True)
        tgz = dl / f"decider-{COMMIT}.tar.gz"
        urllib.request.urlretrieve(f"{REPO}/archive/{COMMIT}.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(dl, filter="data")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# ------------------------------------------------------------------ stage 1: decider's converters, train split only
_load = datasets.load_dataset


def pinned_load(*args, **kw):
    kw["revision"] = REVISIONS[args[0]]
    return _load(*args, **kw)


def stage1(tasks) -> dict[str, list]:
    from decider import data as D
    from decider.data import core

    @core.task("massive_intent_upstream")
    def _massive_upstream():  # decider's massive_intent conversion, on the upstream CC-BY-4.0 repo instead of an unlicensed mirror
        repo, sha = MASSIVE
        tr = _load("parquet", data_files=f"hf://datasets/{repo}@{sha}/en-US/train/0000.parquet", split="train")
        names_all = tr.features["intent"].names
        labs = sorted(set(names_all[i] for i in tr["intent"]))
        idx = {n: i for i, n in enumerate(labs)}
        q = "What is the intent of this voice-assistant utterance?"
        return core._cls(tr, lambda r: r["utt"], lambda r: idx[names_all[r["intent"]]], q, [core._nice(n) for n in labs],
                         "massive_intent", core.TRAIN_CAP), []

    core.load_dataset = datasets.load_dataset = pinned_load  # mind2web imports load_dataset inside its loader
    try:
        pools = {}
        for t in tasks:
            pools[t] = D.load_task(t)[0]
            print(f"decider stage 1: {t}: {len(pools[t])}", flush=True)
        return pools
    finally:
        datasets.load_dataset = _load


# ------------------------------------------------------------------ stage 2
def norm(s):
    return " ".join(s.casefold().split())


def sha(s):
    return hashlib.sha256(norm(s).encode()).hexdigest()


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)


def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from leaves(v)


class Leaks:
    """Raw rows whose text, a line of it, or a 13-word shingle appears in the given eval records."""
    def __init__(self, records):
        self.full, self.ngrams = set(), set()
        for r in records:
            if r.get("_meta", {}).get("text_sha256"):
                self.full.add(r["_meta"]["text_sha256"])
            self.full.add(sha(state_text(r["state"])))
            for s in leaves(r["state"]):
                w = norm(s).split()
                if len(w) >= 4:
                    self.full.add(hashlib.sha256(" ".join(w).encode()).hexdigest())
                self.ngrams.update(" ".join(w[i:i + NG]) for i in range(0, max(0, len(w) - NG + 1)))

    def __call__(self, text):
        w = norm(text).split()
        if hashlib.sha256(" ".join(w).encode()).hexdigest() in self.full:
            return True
        for seg in text.split("\n"):
            seg = re.sub(r"^[A-Za-z0-9 ]{1,30}:\s*", "", seg)
            if seg and sha(seg) in self.full:
                return True
        return any(" ".join(w[i:i + NG]) in self.ngrams for i in range(0, max(0, len(w) - NG + 1)))


def slug(text, used):
    w = re.findall(r"[a-z0-9]+", text.lower())
    w = [x for x in w if x not in ("the", "a", "an", "is", "this", "of", "to", "does", "what", "which", "how", "for", "in", "and")]
    s = "_".join(w[:4]) or "q"
    base, i = s, 2
    while s in used:
        s = f"{base}_{i}"; i += 1
    used.add(s)
    return s


def to_question(q, task_name, src, abstain=None):
    from decider.data import augment as A
    from decider.systemone import strip_level_number
    opts = [o if isinstance(o, tuple) else (o, None) for o in q.options]
    gold = q.gold
    if gold < 0 or gold >= len(opts):
        return None
    names = [n for n, _ in opts]
    plain = all(d is None for _, d in opts)
    if plain and A.is_scale(names) and len(names) >= 3:
        return {"type": "score", "instructions": q.text, "criteria": [strip_level_number(n) for n in names], "label": gold, "src": src}
    if [n.lower() for n in names] == ["no", "yes"] and abstain is None:
        crit = None if plain else {"false": opts[0][1], "true": opts[1][1]}
        if crit is not None and crit["false"] is None and crit["true"] is None:
            crit = None
        q2 = {"type": "noul", "instructions": q.text}
        if crit is not None:
            q2["criteria"] = crit
        q2.update(label=gold, src=src)
        return q2
    if abstain is not None:
        opts = opts + [(abstain, None)]
    letters = (task_name in LETTER_TASKS or len({n.casefold() for n in names}) != len(names) or any(len(n) > 90 for n in names)
               or any(":" in n and plain and len(n) > 40 for n in names))
    if letters:
        keys = [chr(65 + i) for i in range(len(opts))] if len(opts) <= 26 else [str(i + 1) for i in range(len(opts))]
        crit = {k: (n if d is None else f"{n}: {d}" if isinstance(d, str) else {"name": n, **d} if isinstance(d, dict) else n)
                for k, (n, d) in zip(keys, opts)}
    else:
        keys = [n for n, _ in opts]
        crit = {n: d for n, d in opts}
    if len(crit) != len(opts):
        return None
    return {"type": "choice", "instructions": q.text, "criteria": crit, "label": keys[gold], "src": src}


def qdict(qs, task_name, src, abstain=None):
    used, out = set(), {}
    for q in qs:
        jq = to_question(q, task_name, src, abstain if abstain and len(q.options) >= 3 else None)
        if jq is None:
            return None
        out[slug(q.text, used)] = jq
    return out


def make_builder(A, all_train, desc):
    class JBuilder(A.Builder):
        """decider's Builder, with options rendered as (name, description-or-None) pairs instead of 'name: description'."""
        def render(self, opts, mode, ctx=None):
            if mode == "plain":
                return [(o, None) for o, _ in opts]
            r = self.rng
            names = A.opaque_names(len(opts), r) if mode.startswith("opaque") else [o for o, _ in opts]
            keys = r.choice(A.KEYSETS); style = r.randrange(3); out = []
            for (o, k), name in zip(opts, names):
                dd = self.desc.get(k[0], {}).get(k[1], {}).get(o) if k else None
                if dd is None:
                    out.append((o, None) if not mode.startswith("opaque") else (name, o)); continue
                if not mode.startswith("opaque") and r.random() < 0.12:
                    out.append((name, None)); continue
                if mode.endswith("json"):
                    obj = {keys[0]: dd["what"]}
                    if r.random() < 0.6:
                        obj[keys[1]] = dd["not_for"]
                    exs = [x for x in self.by.get(k + (o,), []) if x != ctx]
                    if exs and r.random() < 0.5:
                        obj[keys[2]] = r.sample(exs, min(len(exs), r.randint(1, 3)))
                    if len(obj) == 1 and r.random() < 0.3:
                        obj = obj[keys[0]]
                    out.append((name, obj))
                else:
                    s = (dd["what"] if style == 0 else f"{dd['what']} Not for: {dd['not_for']}" if style == 1
                         else (dd["what"] if r.random() < 0.5 else f"{dd['what']} (not: {dd['not_for']})"))
                    out.append((name, s))
            return out

        def json_state(self, pool, k, n_q, budget_chars, hetero=False):
            """decider's json_state, also recording the records it packed (self.last) for provenance."""
            rng = self.rng
            if hetero:
                recs = rng.sample(pool, k)
            else:
                t = rng.choice(pool).task
                same = [e for e in rng.sample(pool, min(len(pool), 4000)) if e.task == t]
                recs = (same * (k // max(1, len(same)) + 1))[:k] if len(same) < k else rng.sample(same, k)
            out, used = [], 0
            for e in recs:
                if used + len(e.context) > budget_chars and len(out) >= 2:
                    break
                out.append(e); used += len(e.context)
            self.last = out
            # replay the rest of decider's json_state on exactly these records (rng.sample(out, len(out)) is a permutation)
            return A.Builder.json_state(self, out, len(out), n_q, 10**9, hetero=True)

    return JBuilder(all_train, {}, desc, seed=SEED)


def npad(r):  # decider.data.mixture.npad: log-uniform distractor count, heavy tail of very long lists
    return r.randint(100, 250) if r.random() < 0.25 else int(2 ** r.uniform(0, 7.9))


def stage2(raw: dict[str, list], desc: dict, leaks: Leaks) -> list[dict]:
    from decider import data as D
    from decider.data import augment as A
    from decider.data import rules as R
    rng = random.Random(SEED)

    pools, drops = {}, collections.Counter()
    for name in sorted(LICENCE):
        exs = list(raw[name]); random.Random(f"{SEED}:{name}").shuffle(exs)
        keep = []
        for e in exs:
            if len(keep) >= CAP:
                break
            if len(e.context) > MAX_STATE:
                continue
            if leaks(e.context) or sha(e.context) in EXCLUDED_ROWS.get(name, ()):
                drops[name] += 1; continue
            keep.append(e)
        pools[name] = keep
    print(f"decider stage 2: leak drops {dict(drops)}", flush=True)
    task_of = {e.task: n for n, p in pools.items() for e in p}
    B = make_builder(A, [e for p in pools.values() for e in p], desc)

    records, counts, seen, per_source = [], collections.Counter(), set(), collections.Counter()

    def emit(state, qs, source, kind, tasks, aug):
        if isinstance(state, str) and not state.strip():
            return False
        st = state_text(state)
        if len(st) > MAX_STATE or not qs:
            return False
        h = sha(st)
        key = (h, json.dumps([q["instructions"] for q in qs.values()], ensure_ascii=False))
        if key in seen or per_source[source] >= CAP:
            return False
        seen.add(key); per_source[source] += 1
        prov = [PROVENANCE[t] for t in tasks]
        if len({p[0] for p in prov}) == 1:
            repo, rev, lic = prov[0]
        elif prov:
            ps = sorted(set(prov))
            repo, rev, lic = ";".join(p[0] for p in ps), ";".join(p[1] for p in ps), ";".join(sorted({p[2] for p in ps}))
        else:
            repo, rev, lic = "github:Mapika/decider", COMMIT[:7], "apache-2.0"
        meta = {"source": source, "id": f"{source}/{kind}/{per_source[source]:05d}", "text_sha256": h, "repo": repo, "revision": rev,
                "split": "train", "licence": lic, "aug": kind if aug else None, "decider_task": ",".join(sorted(set(tasks))) or kind}
        if not tasks:
            meta["generated"] = True
        records.append({"state": state, "questions": qs, "_meta": meta}); counts[kind] += 1
        return True

    def emit_example(e, name, kind, aug, abstain=None, source=None):
        qs = qdict(e.qs, name, "decider_" + e.task, abstain)
        return qs is not None and emit(e.context, qs, source or f"decider_{name}", kind, [name], aug)

    # assign each raw example one form
    fixed = {k[0] for k, (o, sp) in B.sets.items() if not A.is_scale(o)}
    big = [n for n in pools if pools[n] and pools[n][0].task in BIG_LABEL]
    small = [n for n in pools if pools[n] and pools[n][0].task in fixed and pools[n][0].task not in BIG_LABEL]
    plan = {n: ["plain"] * len(p) for n, p in pools.items()}

    def assign(kind, cands):
        cands = [c for c in cands if plan[c[0]][c[1]] == "plain"]; rng.shuffle(cands)
        for t, i in cands[:BUDGET[kind]]:
            plan[t][i] = kind

    assign("wide", [(n, i) for n in big for i in range(len(pools[n]))])
    assign("padded", [(n, i) for n in small for i, e in enumerate(pools[n]) if all(len(q.options) >= 3 for q in e.qs)])
    assign("described", [(n, i) for n in pools if pools[n][0].task in fixed for i in range(len(pools[n]))])
    assign("single", [(n, i) for n in pools for i, e in enumerate(pools[n]) if len(e.qs) > 1])
    assign("abstain", [(n, i) for n in pools for i, e in enumerate(pools[n])
                       if any(len(q.options) >= 3 and not A.is_scale(q.options) for q in e.qs)])

    for n, p in pools.items():
        for i, e in enumerate(p):
            k = plan[n][i]
            if k == "plain":
                emit_example(A.narrow(e, rng), n, "plain", False)
            elif k == "wide":
                m = max(len(q.options) for q in e.qs); r = rng.random()
                x = B.choice(e, wide=m if r < 0.35 else rng.randint(11, max(11, m)) if r < 0.7 else None,
                             pad=npad(rng) if rng.random() < 0.2 else 0, mode=rng.choice(MODES) if rng.random() < 0.25 else "plain")
                emit_example(x, n, "wide", True)
            elif k == "padded":
                x = B.choice(e, pad=npad(rng), mode=rng.choice(MODES) if rng.random() < 0.3 else "plain")
                emit_example(x, n, "padded", True)
            elif k == "described":
                x = B.choice(A.narrow(e, rng), mode=rng.choice(["named", "named", "opaque", "opaque", "named_json", "opaque_json"]))
                emit_example(x, n, "described", True)
            elif k == "single":
                qs = [rng.choice(e.qs)] if rng.random() < 0.6 else rng.sample(e.qs, rng.randint(1, len(e.qs)))
                x = A.narrow(D.Example(e.context, qs, e.task), rng)
                emit_example(D.Example(x.context, x.qs, e.task + "+fmt"), n, "single", True)
            elif k == "abstain":
                x = A.narrow(e, rng); w = rng.choice(A.ABSTAIN_WORDINGS)
                if rng.random() < A.NONE_GOLD_RATE:  # nothing on offer fits: options from an unrelated task + abstain gold
                    qs = []
                    for q in x.qs:
                        if len(q.options) >= 3 and not A.is_scale(q.options):
                            fam = A.FAM.get(e.task, e.task)
                            others = [t for t in pools if t != n and pools[t][0].task in fixed
                                      and A.FAM.get(pools[t][0].task, pools[t][0].task) != fam]
                            src_t = rng.choice(others)
                            labs = sorted({o for q2 in pools[src_t][0].qs for o in q2.options if not A.is_scale(q2.options)})
                            labs = [o for o in labs if o.casefold() not in {y.casefold() for y in q.options}] or labs
                            opts = rng.sample(labs, min(len(q.options), len(labs))) + [w]
                            qs.append(D.Q(q.text, opts, len(opts) - 1))
                        else:
                            qs.append(q)
                    emit_example(D.Example(x.context, qs, e.task + "+none"), n, "abstain", True)
                else:
                    emit_example(D.Example(x.context, x.qs, e.task + "+none"), n, "abstain", True, abstain=w)

    # JSON states packing several rows
    single = [e for p in pools.values() for e in p if len(e.qs) == 1 and e.qs[0].gold >= 0]
    short = [e for e in single if len(e.context) <= 1200]
    longer = [e for e in single if 300 <= len(e.context) <= 6000]

    def json_emit(e, used, kind):
        st = json.loads(e.context)
        if not isinstance(st, dict):
            return False
        if kind == "json_indexed":
            st = json.loads(A.indexed(D.Example(json.dumps(st, ensure_ascii=False), e.qs, e.task)).context)
        qs = qdict(e.qs, "json", "decider_" + e.task)
        return qs is not None and emit(st, qs, "decider_json_state", kind, sorted({task_of[x.task] for x in used}), True)

    B.rng = rng
    tries = 0
    while counts["json"] < BUDGET["json"] and tries < BUDGET["json"] * 4:
        tries += 1; r = rng.random(); het = rng.random() < 0.4
        if r < 0.10:
            src = rng.choice(single if rng.random() < 0.5 else short); json_emit(B.json_wrap(src), [src], "json")
        elif r < 0.75:
            e = B.json_state(short, rng.randint(2, 10), rng.randint(1, 4), 6000, hetero=het); json_emit(e, B.last, "json")
        else:
            e = B.json_state(longer, rng.randint(4, 24), rng.randint(1, 4), 20000, hetero=het); json_emit(e, B.last, "json")
    tries = 0
    while counts["json_indexed"] < BUDGET["json_indexed"] and tries < BUDGET["json_indexed"] * 4:
        tries += 1; r = rng.random(); het = rng.random() < 0.4
        e = (B.json_state(short, rng.randint(8, 40), rng.randint(1, 4), 12000, hetero=het) if r < 0.55 else
             B.json_state(longer, rng.randint(8, 30), rng.randint(1, 4), 20000, hetero=het) if r < 0.8 else
             B.json_state(short, rng.randint(2, 7), rng.randint(1, 3), 6000, hetero=het))
        json_emit(e, B.last, "json_indexed")

    # isolated: one yes/no row per level of a Score question (scale tasks) or per option of a small Choice question
    iso_scale = [(n, e, q) for n, p in pools.items() for e in p if e.task in SCALE_TASKS for q in e.qs
                 if len(q.options) >= 3 and A.is_scale(q.options) and q.gold >= 0]
    iso_choice = [(n, e, q) for n, p in pools.items() for e in p if e.task not in SCALE_TASKS and len(e.context) < 1500 for q in e.qs
                  if 3 <= len(q.options) <= 8 and not A.is_scale(q.options) and q.gold >= 0]
    rng.shuffle(iso_scale); rng.shuffle(iso_choice)
    for lst, share in ((iso_scale, 0.75), (iso_choice, 0.25)):
        target = counts["isolated"] + int(BUDGET["isolated"] * share)
        for n, e, q in lst:
            if counts["isolated"] >= target:
                break
            for x in A.isolated(e, q, e.task + "+iso"):
                emit_example(x, n, "isolated", True, source="decider_isolated")

    # rule-conditioned decisions with state twins and rule twins (programmatic labels)
    for x in R.build(BUDGET["rules"] + 500, seed=SEED + 11):
        if counts["rules"] >= BUDGET["rules"]:
            break
        st = x.context
        try:
            j = json.loads(st)
            if isinstance(j, dict):
                st = j
        except Exception:
            pass
        qs = qdict(x.qs, "rules", "decider_" + x.task)
        if qs is not None:
            emit(st, qs, "decider_rules", "rules", [], True)

    # keep augmented rows to at most AUG_MAX_FRAC of the output
    aug = [r for r in records if r["_meta"]["aug"]]; base = [r for r in records if not r["_meta"]["aug"]]
    lim = int(AUG_MAX_FRAC * len(base) / (1 - AUG_MAX_FRAC))
    if len(aug) > lim:
        rng.shuffle(aug); aug = aug[:lim]
    out = base + aug; rng.shuffle(out)
    print(f"decider stage 2: {len(out)} rows, {len(aug)} augmented, {dict(counts)}", flush=True)
    return out


def build(dl: Path, eval_records: list[dict]) -> list[dict]:
    """decider_public rows. eval_records: the Jeeves dev/test records the raw rows are decontaminated against."""
    root = fetch(dl)
    raw = stage1(sorted(LICENCE))
    desc = json.loads((root / "teacher_data" / "label_descriptions.json").read_text())
    return stage2(raw, desc, Leaks(eval_records))
