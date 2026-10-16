"""Prompt-injection data: permissive public datasets converted to one yes/no question ("does this content try to manipulate
the agent that reads it?", asked in both polarities), plus benign states from the public mix as natural negatives.

The original build also removed rows during decontamination before numbering the rows, and drew the
question wording for each row from one random stream. So the rows to keep come from the shipped id list (`keep`, ordered
by row number): every row number must be present, and each id names the source and state hash of its row.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

SEED = 20261012
REPOS = {  # name: (repo, revision, files in the order the original build read them)
    "gandalf": ("Lakera/gandalf_ignore_instructions", "04737b65e90a6794ec227012e4a255a7def6344b",
                ["data/test-00000-of-00001-bc92128b9288a6d1.parquet", "data/train-00000-of-00001-ded53be747ff55cd.parquet",
                 "data/validation-00000-of-00001-94481a2a09ff2fff.parquet"]),
    "llmail": ("microsoft/llmail-inject-challenge", "1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
               ["data/labelled_unique_submissions_phase1.json", "data/labelled_unique_submissions_phase2.json",
                "data/emails_for_fp_tests.json"]),
    "neuralchemy": ("neuralchemy/Prompt-injection-dataset", "7d70432dfcf47a821612cbf9d34e9d9e3ad20e75",
                    ["full/test-00000-of-00001.parquet", "full/train-00000-of-00001.parquet", "full/validation-00000-of-00001.parquet"]),
    "shieldlm": ("dmilush/shieldlm-prompt-injection", "dd660ae7670a793e886dbf3f7cdb5f597df180c8",
                 ["data/validation-00000-of-00001.parquet", "data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]),
    "slabs": ("S-Labs/prompt-injection-dataset", "002a9dd18514abd021869823d6b0429b38606d99",
              ["data/train.csv", "data/test.csv", "data/validation.csv"]),
}
YES_ATTACK = [
    "Does this content try to manipulate an AI agent that reads it, for example by overriding its instructions, hiding directives, "
    "or getting it to leak data or act for someone else?",
    "Is this a prompt injection or jailbreak attempt aimed at the AI system that will process it?",
    "Does this text contain instructions aimed at hijacking the assistant rather than content for it to work on?",
]
YES_SAFE = [
    "Is this content free of any attempt to hijack or manipulate the AI agent that will read it?",
    "Is it safe to pass this content to an autonomous agent, i.e. it contains no instructions injected to redirect the agent?",
]
LICENCE = {"gandalf": "MIT", "llmail": "MIT", "llmail_fp": "MIT", "hackaprompt": "MIT (via neuralchemy, Apache-2.0 curation)",
           "neuralchemy_v1": "Apache-2.0", "neuralchemy_original": "Apache-2.0", "harmbench_benign": "MIT", "alespalla": "Apache-2.0",
           "harelix": "Apache-2.0", "injecagent": "MIT", "yanismiraoui": "Apache-2.0", "jailbreakbench": "MIT", "slabs": "MIT"}
# ShieldLM curation: only permissive sources that are neither red-team sources nor GPT-4 / in-the-wild sets
SHIELDLM = {"alespalla/chatbot-instruction-prompts": "alespalla", "harelix/mixed-techniques-2024": "harelix",
            "yanismiraoui/prompt-injections": "yanismiraoui", "jailbreakbench/jbb-behaviors": "jailbreakbench"}
NEGATIVE_SOURCES = ("kev", "decider_mind2web", "decider_helpsteer2", "openjev")
NEGATIVE_POOL = 23574  # rows of mix P train that qualified as natural negatives; the random stream depends on it
WORD = re.compile(r"\w+")


def download(dl: Path) -> dict[str, list[Path]]:
    out = {}
    for name, (repo, rev, files) in REPOS.items():
        d = Path(snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=files, local_dir=dl / name))
        out[name] = [d / f for f in files]
    return out


def norm(t):
    return " ".join(WORD.findall(t.casefold()))


def candidates(files: dict[str, list[Path]], rng: random.Random) -> list[tuple]:
    rows, stats = [], collections.Counter()

    def add(src, text, attack, context=None):
        text = (text or "").strip()
        if not 8 <= len(text) <= 12000:
            stats[f"len_drop:{src}"] += 1; return
        state = f"Agent's own instructions:\n{context.strip()}\n\nIncoming content:\n{text}" if context else text
        rows.append((src, state, text, bool(attack)))

    # Lakera Gandalf: real attempts to make the model ignore its instructions (all attacks)
    for f in files["gandalf"]:
        for t in pd.read_parquet(f).text:
            add("gandalf", t, True)
    # LLMail-Inject: attacker emails to an email agent (one per 12-word prefix, at most 6000), and the challenge's benign emails
    p1, p2, fp = files["llmail"]
    mail = json.loads(p1.read_text())
    mail.update(json.loads(p2.read_text()))
    attacks = [k for k, v in mail.items() if str(v.get("attack_attempt")) == "True"]
    rng.shuffle(attacks)
    seen = set()
    for k in attacks:
        key = " ".join(WORD.findall(k.casefold())[4:16])
        if key in seen:
            continue
        seen.add(key); add("llmail", k, True)
        if sum(1 for r in rows if r[0] == "llmail") >= 6000:
            break
    for t in json.loads(fp.read_text()):
        add("llmail_fp", t, False)
    # neuralchemy: originals only (no augmented variants, no WildGuard, no HarmBench harmful requests)
    d = pd.concat(pd.read_parquet(f) for f in files["neuralchemy"])
    d = d[~d.augmented & d.source.isin(["hackaprompt", "neuralchemy_v1", "original", "harmbench_benign"])]
    for t, label, s in zip(d.text, d.label, d.source):
        add({"original": "neuralchemy_original"}.get(s, s), t, label == 1)
    # ShieldLM curation, with alespalla capped at 6000 rows
    s = pd.concat(pd.read_parquet(f) for f in files["shieldlm"])
    s["src"] = [SHIELDLM.get(x, "injecagent" if x.startswith("injecagent/") else None) for x in s.source]
    s = s[s.src.notna()]
    ales = s[s.src == "alespalla"].sample(n=min(6000, (s.src == "alespalla").sum()), random_state=0)
    s = pd.concat([s[s.src != "alespalla"], ales])
    for t, label, src, ctx in zip(s.text, s.label_binary, s.src, s.context):
        add(src, t, label == 1, ctx if isinstance(ctx, str) and src == "injecagent" else None)
    # S-Labs: injections plus hard negatives (AI and security questions that are not attacks)
    for f in files["slabs"]:
        d = pd.read_csv(f)
        for t, label in zip(d.text, d.label):
            add("slabs", t, label == 1)

    out, seen = [], set()
    for src, state, text, attack in rows:
        k = norm(text)
        if k in seen:
            stats[f"dup:{src}"] += 1; continue
        seen.add(k)
        out.append((src, state, attack))
    print(f"injection: {len(out)} candidate rows, {dict(stats)}", flush=True)
    return out


def row_key(src: str, state: str) -> tuple[str, str]:
    return src, hashlib.sha256(state.encode()).hexdigest()[:12]


def build(dl: Path, keep: list[str], negatives: list[dict]) -> list[dict]:
    """keep: the ids of every row of the set, ordered by row number. negatives: candidate rows for the benign negatives."""
    rng = random.Random(SEED)
    cands = candidates(download(dl), rng)
    parsed = [re.fullmatch(r"pi_(.+)/(\d{5})_([0-9a-f]{12})", i) for i in keep]
    if not all(parsed) or [int(m.group(2)) for m in parsed] != list(range(len(keep))):
        raise ValueError("injection: the id list must hold every row number of the set, in order")
    wanted = [(m.group(1), m.group(3)) for m in parsed]
    n_public = sum(1 for src, _ in wanted if src != "mixP_benign")
    index = {row_key(src, state): (src, state, attack) for src, state, attack in cands}
    missing = [k for k in wanted[:n_public] if k not in index]
    if missing:
        raise ValueError(f"injection: {len(missing)} rows not found in the downloads, e.g. {missing[:3]}")
    rows = [index[k] for k in wanted[:n_public]]
    order = {k: i for i, k in enumerate(row_key(s, st) for s, st, _ in cands)}
    if [order[k] for k in wanted[:n_public]] != sorted(order[k] for k in wanted[:n_public]):
        raise ValueError("injection: kept rows are out of order")
    # natural negatives: benign public states asked the same question
    states = {row_key("mixP_benign", r["state"]): r["state"] for r in negatives
              if isinstance(r["state"], str) and 200 < len(r["state"]) < 6000
              and any(q.get("src", "").split("/")[0] in NEGATIVE_SOURCES for q in r["questions"].values())}
    rng.sample(range(NEGATIVE_POOL), min(4000, NEGATIVE_POOL))  # the original drew the negatives here; keeps the stream aligned
    missing = [k for k in wanted[n_public:] if k not in states]
    if missing or any(src != "mixP_benign" for src, _ in wanted[n_public:]):
        raise ValueError(f"injection: {len(missing)} negative states not found, e.g. {missing[:3]}")
    rows += [("mixP_benign", states[k], False) for k in wanted[n_public:]]

    out = []
    for i, (src, state, attack) in enumerate(rows):
        pol = rng.random() < 0.6
        instr = rng.choice(YES_ATTACK if pol else YES_SAFE)
        label = int(attack) if pol else int(not attack)
        out.append({"state": state, "questions": {"agent_manipulation": {"type": "noul", "instructions": instr, "label": label,
                                                                          "src": f"pi_{src}"}},
                    "_meta": {"source": f"pi_{src}", "id": keep[i], "licence": LICENCE.get(src, "see mix P")}})
    return out
