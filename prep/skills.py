"""General-skill datasets (numbers, commonsense, tool choice, fact checking, retrieval, dialogue) converted from their train
splits. Questions are fixed templates; options and labels come from the datasets (no model-written text). Each dataset is
sampled with random.Random("ext2:<name>") and a deterministic 5% hold-out of it goes to dev.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import re
import string
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

MAX_STATE = 6000
SOURCES = {  # name: (repo, revision, licence, files)
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "mit", ["main/train-00000-of-00001.parquet"]),
    "aqua_rat": ("deepmind/aqua_rat", "33301c6a050c96af81f63cad5562cb5363e88971", "apache-2.0", ["raw/train-00000-of-00001.parquet"]),
    "ruletaker": ("tasksource/ruletaker", "a3e0880baeb6ec3d478f4c4d85afe04b21b6cf7f", "apache-2.0",
                  ["data/train-00000-of-00001-52adaa842dd7ed92.parquet"]),
    # HF card states no licence; the upstream GitHub repo is Apache-2.0
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "apache-2.0",
                   ["winogrande_xl/train-00000-of-00001.parquet"]),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "mit", ["data/train-00000-of-00001.parquet"]),
    # HF card says "unknown"; the upstream GitHub repo is MIT
    "swag": ("allenai/swag", "dc48148372b3853a9c7bae7bb06c161b46d8364a", "mit", ["regular/train-00000-of-00001.parquet"]),
    "cosmos_qa": ("allenai/cosmos_qa", "ed50fe83db62b355759053b1cd5eef344ee371e0", "cc-by-4.0", ["default/train/0000.parquet"]),
    # HF card says "unknown"; the upstream GitHub repo is Apache-2.0
    "openbookqa": ("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", "apache-2.0", ["main/train-00000-of-00001.parquet"]),
    "newyorker": ("jmhessel/newyorker_caption_contest", "d81cbab7d0392708d5371d3a4960e69261824db4", "cc-by-4.0",
                  ["matching/train-00000-of-00002.parquet", "matching/train-00001-of-00002.parquet"]),
    "when2call": ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", "cc-by-4.0", ["train/when2call_train_pref.jsonl"]),
    "toolace": ("Team-ACE/ToolACE", "6bda777c88d21e5a204703c1ee45597a8fa4f734", "apache-2.0", ["data.json"]),
    "glaive_fc": ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac", "apache-2.0",
                  ["glaive-function-calling-v2.json"]),
    # the original RAGTruth release (github ParticleMedia/RAGTruth, MIT)
    "ragtruth": ("jakobsnel/RAGTruth_Xtended", "66953ac1f7bf8bc997a41ba96fc7323864c0a616", "mit",
                 ["RAGTruth/response.jsonl", "RAGTruth/source_info.jsonl"]),
    "esci": ("tasksource/esci", "8113b17a5d4099e20243282c926f1bc1a08a4d13", "apache-2.0",
             ["data/train-00000-of-00011-2d36455632bef8a2.parquet", "data/train-00001-of-00011-18b81793a483996f.parquet",
              "data/train-00002-of-00011-71f741fdff9a6f54.parquet"]),
    # revision is the refs/convert/parquet commit
    "sgd": ("google-research-datasets/schema_guided_dstc8", "a8029ad8c12e13cdbfefc2f649e245fa135bc98a", "cc-by-sa-4.0",
            ["dialogues/train/0000.parquet", "schema/train/0000.parquet"]),
}
LETTERS = "ABCDEFGH"


def download(dl: Path, name: str) -> list[Path]:
    repo, rev, _, files = SOURCES[name]
    return [Path(hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=dl / name)) for f in files]


def record(name, rid, state, questions):
    repo, rev, lic, _ = SOURCES[name]
    for q in questions.values():
        q["src"] = name
    return {"state": state, "questions": questions,
            "_meta": {"source": name, "id": f"{name}/train/{rid}", "repo": repo, "revision": rev, "split": "train", "licence": lic}}


def lowest(keys, n):
    """The n keys with the lowest sha256: the dev hold-out."""
    return set(sorted(keys, key=lambda k: hashlib.sha256(str(k).encode()).hexdigest())[:n])


# ================================================================ numbers and reasoning
def fmt_num(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def threshold_question(rng, value, templates):
    """Yes/no 'greater than T' question, T below or above the true value with equal probability."""
    yes = rng.random() < 0.5
    gap = max(1.0, abs(value) * rng.uniform(0.1, 0.5))
    if value == int(value) and abs(value) >= 1:
        gap = max(1, round(gap))
    t = value - gap if yes else value + gap
    t = round(t) if abs(t) >= 10 else round(t, 2)
    if (t < value) != yes:  # rounding pushed it across; nudge back
        t = value - 1 if yes else value + 1
    return {"type": "noul", "instructions": rng.choice(templates).format(t=fmt_num(t)), "label": int(yes)}


GSM_CHOICE = ["What is the answer to this problem?", "Which of these is the correct answer?",
              "Pick the right final answer.", "Which number solves the problem?"]
GSM_NOUL = ["Is the answer greater than {t}?", "Does the answer come to more than {t}?", "Is the final number above {t}?"]


def gsm_distractors(rng, ans, solution):
    """Wrong options: intermediate results from the reference solution, then simple numeric rules."""
    seen, out = {ans}, []
    inter = [float(v.replace(",", "")) for v in re.findall(r"<<[^=>]*=(-?[\d.,]+)>>", solution)
             if re.fullmatch(r"-?\d[\d,]*(\.\d+)?", v)]
    rng.shuffle(inter)
    # all rule-based distractors go one way (picked per item), so the answer is never the midpoint of a +/- pair
    s = 1 if rng.random() < 0.7 else -1  # mostly up: intermediate results already sit below the answer
    rules = [ans + s, ans + 5 * s, ans + 10 * s, ans * 2 ** s, ans * 1.5 ** s, ans * 10 ** s]
    rng.shuffle(rules)
    for v in inter + rules:
        v = round(v, 2)
        if ans.is_integer() and not float(v).is_integer():
            continue
        if v not in seen and (v >= 0 or ans < 0) and len(out) < 3:
            seen.add(v)
            out.append(v)
    return out


def gsm8k(df, rng):
    rows = []
    for i, r in df.iterrows():
        a = r["answer"].split("####")[-1].strip().replace(",", "")
        if not re.fullmatch(r"-?\d+(\.\d+)?", a) or len(r["question"]) > MAX_STATE:
            continue
        ans = float(a)
        if rng.random() < 0.6:
            opts = [ans] + gsm_distractors(rng, ans, r["answer"])
            rng.shuffle(opts)
            keys = "ABCD"[:len(opts)]
            q = {"type": "choice", "instructions": rng.choice(GSM_CHOICE),
                 "criteria": {k: fmt_num(v) for k, v in zip(keys, opts)}, "label": keys[opts.index(ans)]}
            rows.append(record("gsm8k", i, r["question"], {"final_answer": q}))
        else:
            rows.append(record("gsm8k", i, r["question"], {"answer_above_threshold": threshold_question(rng, ans, GSM_NOUL)}))
    return rows


AQUA_CHOICE = ["Which option is correct?", "Which of these answers is right?", "Pick the correct answer."]
AQUA_NOUL = ["Is the answer {o}?", "Would {o} be the correct answer?", "Is {o} right?"]
POSITIONAL = re.compile(r"\babove\b|\bboth\b|\b[A-E] (and|or|&) [A-E]\b|\boption\b", re.I)


def aqua_rat(df, rng):
    rows, seen_q = [], set()
    for i, r in df.iterrows():
        opts = [re.sub(r"^[A-E]\)", "", o).strip() for o in r["options"]]
        if (r["question"] in seen_q or len(r["question"]) > MAX_STATE or r["correct"] not in "ABCDE"
                or len(set(opts)) != 5 or any(not o or POSITIONAL.search(o) for o in opts)):
            continue
        seen_q.add(r["question"])
        right = opts["ABCDE".index(r["correct"])]
        if rng.random() < 0.75:
            rng.shuffle(opts)  # dataset order is skewed (E is rarely right), so reshuffle
            q = {"type": "choice", "instructions": rng.choice(AQUA_CHOICE),
                 "criteria": dict(zip("ABCDE", opts)), "label": "ABCDE"[opts.index(right)]}
            rows.append(record("aqua_rat", i, r["question"], {"option_correct": q}))
        else:
            yes = rng.random() < 0.5
            o = right if yes else rng.choice([x for x in opts if x != right])
            q = {"type": "noul", "instructions": rng.choice(AQUA_NOUL).format(o=o), "label": int(yes)}
            rows.append(record("aqua_rat", i, r["question"], {"answer_is_option": q}))
    return rows


RT_NOUL = ["Based only on these facts and rules, is the following statement true? \"{s}\"",
           "Going strictly by the facts and rules given, does it follow that: {s}",
           "Can we conclude \"{s}\" from the rules and facts above?"]


def ruletaker(df, rng):
    rows = []
    for i, r in df.drop_duplicates("context").iterrows():  # one question per context, so no context spans train and dev
        if r["label"] not in ("entailment", "not entailment") or len(r["context"]) > MAX_STATE:
            continue
        q = {"type": "noul", "instructions": rng.choice(RT_NOUL).format(s=r["question"]), "label": int(r["label"] == "entailment")}
        rows.append(record("ruletaker", i, r["context"], {"statement_follows": q}))
    return rows


WG_CHOICE = ["Who or what does the blank (_) refer to?", "Which option best fills the blank?",
             "Which of these makes the most sense in place of the _?"]


def winogrande(df, rng):
    rows = []
    for i, r in df.drop_duplicates("sentence").iterrows():
        if r["answer"] not in ("1", "2") or r["sentence"].count("_") != 1:
            continue
        opts = [r["option1"], r["option2"]]
        right = opts[int(r["answer"]) - 1]
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": rng.choice(WG_CHOICE), "criteria": dict(zip("AB", opts)), "label": "AB"[opts.index(right)]}
        rows.append(record("winogrande", i, r["sentence"], {"fill_blank": q}))
    return rows


REASONING = {"gsm8k": (gsm8k, 4000), "aqua_rat": (aqua_rat, 4000), "ruletaker": (ruletaker, 3000), "winogrande": (winogrande, 3000)}


def reasoning(name, files):
    """Sample rows first (by row index), then convert, so the cap counts questions actually written; dev = lowest 5% by id."""
    fn, cap = REASONING[name]
    rng = random.Random(f"ext2:{name}")
    df = pd.read_parquet(files[0])
    idx = list(df.index)
    rng.shuffle(idx)
    rows = fn(df.loc[idx[:3 * cap]], rng)[:cap]
    dev = lowest([r["_meta"]["id"] for r in rows], math.ceil(0.05 * len(rows)))
    return [r for r in rows if r["_meta"]["id"] not in dev], [r for r in rows if r["_meta"]["id"] in dev]


# ================================================================ commonsense and plausibility
CAP = 3000


def choice_q(rng, templates, options, correct, shuffle=True):
    order = list(range(len(options)))
    if shuffle:
        rng.shuffle(order)
    return {"type": "choice", "instructions": rng.choice(templates), "criteria": {LETTERS[k]: options[i] for k, i in enumerate(order)},
            "label": LETTERS[order.index(correct)]}


def by_group(name, items):
    """items: (record id, group id, state, question name, question); dev = the lowest 5% of the group ids."""
    groups = {g for _, g, *_ in items}
    dev = lowest(groups, max(1, round(len(groups) * 0.05)))
    rows = {"train": [], "dev": []}
    for rid, gid, state, qname, q in items:
        rows["dev" if gid in dev else "train"].append(record(name, rid, state, {qname: q}))
    return rows["train"], rows["dev"]


WIKIHOW_TAG = re.compile(r"\s*\[(header|title|step|substeps)\]\s*")


def hellaswag(files):
    rng = random.Random("ext2:hellaswag")
    rows = [r for r in pd.read_parquet(files[0]).itertuples() if r.label in {"0", "1", "2", "3"}]
    rng.shuffle(rows)
    templates = ["Which ending most plausibly continues this?", "What is most likely to happen next?",
                 "Pick the most sensible continuation of this description."]
    items = []
    for r in rows:
        state = f"Activity: {r.activity_label}\n{WIKIHOW_TAG.sub(' ', r.ctx).strip()} ..."
        if len(state) > MAX_STATE:
            continue
        opts = [WIKIHOW_TAG.sub(" ", e).strip() for e in r.endings]
        items.append((r.ind, r.ind, state, "next_event", choice_q(rng, templates, opts, int(r.label), shuffle=False)))
        if len(items) == CAP:
            break
    return by_group("hellaswag", items)


def swag(files):
    rng = random.Random("ext2:swag")
    rows = list(pd.read_parquet(files[0]).itertuples())
    rng.shuffle(rows)
    templates = ["Which ending most plausibly follows?", "What most likely happens next?", "Pick the most sensible way to finish this sentence."]
    items = []
    for r in rows:
        opts = [getattr(r, f"ending{i}").strip() for i in range(4)]  # dataset order is already random
        items.append((r.Index, r.Index, f"{r.sent1.strip()} {r.sent2.strip()} ...", "next_event",
                      choice_q(rng, templates, opts, int(r.label), shuffle=False)))
        if len(items) == CAP:
            break
    return by_group("swag", items)


def openbookqa(files):
    rng = random.Random("ext2:openbookqa")
    rows = list(pd.read_parquet(files[0]).itertuples())
    rng.shuffle(rows)
    templates = ["Which answer is correct?", "Which option correctly completes or answers this?", "Choose the correct answer."]
    items = []
    for r in rows:
        labels, texts = list(r.choices["label"]), [t.strip() for t in r.choices["text"]]
        if r.answerKey not in labels:
            continue
        items.append((r.id, r.id, r.question_stem.strip(), "correct_answer", choice_q(rng, templates, texts, labels.index(r.answerKey))))
        if len(items) == CAP:
            break
    return by_group("openbookqa", items)


def cosmos_qa(files):
    rng = random.Random("ext2:cosmos_qa")
    rows = list(pd.read_parquet(files[0]).itertuples())
    rng.shuffle(rows)
    templates = ["Which answer is most plausible given the passage?", "Based on the text, which option best answers the question?",
                 "Choose the most reasonable answer to the question about this passage."]
    items = []
    for r in rows:
        state = f"{r.context.strip()}\n\nQuestion: {r.question.strip()}"
        if len(state) > MAX_STATE:
            continue
        opts = [getattr(r, f"answer{i}").strip() for i in range(4)]
        items.append((r.id, r.id, state, "plausible_answer", choice_q(rng, templates, opts, int(r.label))))
        if len(items) == CAP:
            break
    return by_group("cosmos_qa", items)


def newyorker(files, per_contest=7):
    """Caption matching, text fields only. At most 7 instances per contest; the hold-out is by contest."""
    rng = random.Random("ext2:newyorker")
    cols = ["contest_number", "image_location", "image_description", "image_uncanny_description", "questions", "caption_choices",
            "label", "instance_id"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    rows = sorted(df.itertuples(), key=lambda r: (r.contest_number, r.instance_id))
    rng.shuffle(rows)
    templates = ["Which caption was written for this cartoon?", "Which of these captions matches the cartoon described?",
                 "Pick the caption that belongs to this cartoon."]
    items, per = [], collections.Counter()
    for r in rows:
        if per[r.contest_number] >= per_contest:
            continue
        lines = [f"Cartoon setting: {r.image_location}", f"What it shows: {r.image_description}",
                 f"What is unusual: {r.image_uncanny_description}"] + [f"Question the scene raises: {x}" for x in r.questions]
        state = "\n".join(lines)
        if len(state) > MAX_STATE:
            continue
        q = choice_q(rng, templates, list(r.caption_choices), LETTERS.index(r.label))
        items.append((f"{r.contest_number}_{r.instance_id}", r.contest_number, state, "matching_caption", q))
        per[r.contest_number] += 1
        if len(items) == CAP:
            break
    return by_group("newyorker", items)


# ================================================================ tool choice, fact checking, retrieval, dialogue
MAX_Q = 3000


def capped(name, items):
    """items: (id, state, questions) in sampled order. Caps questions, then holds out the lowest 5% of the ids by sha256."""
    picked, nq = [], 0
    for it in items:
        if len(it[1]) > MAX_STATE:
            continue
        if nq + len(it[2]) > MAX_Q:
            break
        picked.append(it)
        nq += len(it[2])
    dev = set(sorted((it[0] for it in picked), key=lambda oid: hashlib.sha256(f"{name}/train/{oid}".encode()).hexdigest())
              [: round(len(picked) * 0.05)])
    rows = {"train": [], "dev": []}
    for oid, state, qs in picked:
        rows["dev" if oid in dev else "train"].append(record(name, oid, state, qs))
    return rows["train"], rows["dev"]


def q_choice(instructions, criteria, label):
    return {"type": "choice", "instructions": instructions, "criteria": criteria, "label": label}


def q_noul(instructions, label):
    return {"type": "noul", "instructions": instructions, "label": int(label)}


def lettered(r, options, correct):
    """Options shuffled under letters A, B, ...; returns the criteria and the correct letter."""
    order = list(range(len(options)))
    r.shuffle(order)
    return {string.ascii_uppercase[i]: options[j] for i, j in enumerate(order)}, string.ascii_uppercase[order.index(correct)]


def render_tools(tools):
    lines = []
    for i, t in enumerate(tools, 1):
        params = t.get("parameters") if isinstance(t.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        req = set(params["required"]) if isinstance(params.get("required"), list) else set()
        lines.append(f"{i}. {t['name']}: {t.get('description', '').strip()}")
        for p, spec in props.items():
            spec = spec if isinstance(spec, dict) else {}
            lines.append(f"    - {p} ({spec.get('type', '')}{', required' if p in req else ''}): {spec.get('description', '')}".rstrip(": "))
    return "\n".join(lines) if lines else "(no tools available)"


def tool_state(tools, request):
    return f"Available tools:\n{render_tools(tools)}\n\nUser request:\n{request.strip()}"


TOOL_TEMPLATES = ["Which tool should the assistant call for this request, if any?", "What should the assistant do next with this request?",
                  "Given these tools, what is the right next step for the assistant?"]


def tool_question(r, tools, kind, called=None, extra=()):
    """One option per tool plus the non-call options in extra; kind is 'call' or a key of extra."""
    crit = {f"call_{i}": f"Call {t['name']}" for i, t in enumerate(tools, 1)}
    crit.update(dict(extra))
    label = next(k for k, t in zip(crit, tools) if t["name"] == called) if kind == "call" else kind
    return q_choice(r.choice(TOOL_TEMPLATES), crit, label)


def when2call(files):
    """Label from the chosen response: a tool call, a question back to the user, or a refusal."""
    r = random.Random("ext2:when2call")
    extra = [("ask_user", "Ask the user for the missing information"),
             ("cannot_help", "Tell the user it can't do this with the available tools")]
    items = []
    for i, row in enumerate(map(json.loads, open(files[0]))):
        tools = [json.loads(t) for t in row["tools"]]
        ch = row["chosen_response"]["content"]
        if ch.startswith("<TOOLCALL>"):
            calls = json.loads(ch[len("<TOOLCALL>"):-len("</TOOLCALL>")])
            if len({c["name"] for c in calls}) != 1 or calls[0]["name"] not in [t["name"] for t in tools]:
                continue
            kind, called = "call", calls[0]["name"]
        elif "?" in ch or re.search(r"\b(I need|I'll need|please (specify|provide))\b", ch, re.I):
            kind, called = "ask_user", None
        elif re.search(r"unable|can't|cannot|not able|don't have", ch, re.I):
            kind, called = "cannot_help", None
        else:
            continue
        if len(tools) > 8:
            continue
        items.append((i, tool_state(tools, row["messages"][-1]["content"]), {"next_action": tool_question(r, tools, kind, called, extra)}))
    r.shuffle(items)
    return capped("when2call", items)


def toolace(files):
    """Label from the first assistant turn: '[Tool(...)]' calling exactly one listed tool, or plain text (no_call)."""
    r = random.Random("ext2:toolace")
    marker = "Here is a list of functions in JSON format that you can invoke:\n"
    extra = [("no_call", "Don't call a tool; reply to the user directly (e.g. ask for missing details or explain it can't be done)")]
    items = []
    for i, row in enumerate(json.load(open(files[0]))):
        conv = row["conversations"]
        try:
            tools = json.JSONDecoder().raw_decode(row["system"].split(marker, 1)[1].strip())[0]
        except (IndexError, ValueError):
            continue
        if conv[0]["from"] != "user" or conv[1]["from"] != "assistant" or not 1 <= len(tools) <= 8:
            continue
        names = [t["name"] for t in tools]
        resp = conv[1]["value"].strip()
        called = {n for n in names if re.search(r"(^\[|,\s*)" + re.escape(n) + r"\(", resp)} if resp.startswith("[") else None
        if called is None:
            kind, c = "no_call", None
        elif len(called) == 1:
            kind, c = "call", called.pop()
        else:
            continue
        items.append((i, tool_state(tools, conv[0]["value"]), {"next_action": tool_question(r, tools, kind, c, extra)}))
    r.shuffle(items)
    return capped("toolace", items)


def glaive_fc(files):
    """Label: the first <functioncall> before the user speaks again, else no_call."""
    r = random.Random("ext2:glaive_fc")
    extra = [("no_call", "Don't call a function; reply to the user directly")]
    items = []
    for i, row in enumerate(json.load(open(files[0]))):
        sysm = row["system"]
        if "with access to the following functions" not in sysm:
            continue
        text, tools, dec = sysm.split("Use them if required -", 1)[-1], [], json.JSONDecoder()
        try:
            text = text.strip()
            while text:
                obj, end = dec.raw_decode(text)
                tools.append(obj)
                text = text[end:].strip()
        except ValueError:
            continue
        turns = re.split(r"\n*(USER|ASSISTANT|FUNCTION RESPONSE): ", "\n" + row["chat"])
        turns = [(turns[k], turns[k + 1].replace("<|endoftext|>", "").strip()) for k in range(1, len(turns) - 1, 2)]
        if not turns or turns[0][0] != "USER" or not 1 <= len(tools) <= 8 or not all(isinstance(t, dict) and t.get("name") for t in tools):
            continue
        call = None
        for role, txt in turns[1:]:
            if role == "USER":
                break
            m = re.search(r'<functioncall>\s*\{"name":\s*"([^"]+)"', txt)
            if m:
                call = m.group(1)
                break
        if call is not None and call not in [t.get("name") for t in tools]:
            continue
        items.append((i, tool_state(tools, turns[0][1]),
                      {"next_action": tool_question(r, tools, "call" if call else "no_call", call, extra)}))
    r.shuffle(items)
    return capped("glaive_fc", items)


RT_TYPES = {"Evident Conflict": "It contradicts the source in an obvious way",
            "Subtle Conflict": "It subtly distorts or contradicts the source",
            "Evident Baseless Info": "It adds information clearly not found in the source",
            "Subtle Baseless Info": "It adds minor details or embellishments not found in the source"}


def ragtruth(files):
    """Train rows of quality 'good'. noul: any annotated hallucination span (asked in either polarity); hallucination_type when
    all spans of a response share one type."""
    r = random.Random("ext2:ragtruth")
    response, source_info = files
    src = {s["source_id"]: s for s in map(json.loads, open(source_info))}
    items = []
    for row in map(json.loads, open(response)):
        if row["split"] != "train" or row["quality"] != "good":
            continue
        s = src[row["source_id"]]
        state = f"Task ({s['task_type']}, source {s['source']}):\n{s['prompt'].strip()}\n\nResponse:\n{row['response'].strip()}"
        bad = bool(row["labels"])
        if r.random() < 0.5:
            qs = {"unsupported_claims": q_noul(r.choice(["Does this response contain claims not supported by the passage?",
                                                         "Does the response say anything the source material doesn't back up?"]), bad)}
        else:
            qs = {"fully_grounded": q_noul(r.choice(["Is everything in this response supported by the source material?",
                                                     "Does the response stick strictly to what the source says?"]), not bad)}
        types = {l["label_type"] for l in row["labels"]}
        if len(types) == 1:
            crit, lab = lettered(r, list(RT_TYPES.values()), list(RT_TYPES).index(types.pop()))
            qs["hallucination_type"] = q_choice("What kind of problem does the unsupported part of this response have?", crit, lab)
        items.append((row["id"], state, qs))
    r.shuffle(items)
    return capped("ragtruth", items)


ESCI = {"Exact": "Exact match: the product is relevant for the query and meets all its specifications",
        "Substitute": "Substitute: somewhat relevant; it fails some aspects of the query but could work as a substitute",
        "Complement": "Complement: it doesn't fit the query but could be used together with an exact item",
        "Irrelevant": "Irrelevant: it fails to fulfil a central aspect of the query"}


def esci(files):
    """us-locale rows of the first 3 of 11 train shards, up to 750 per label."""
    r = random.Random("ext2:esci")
    cols = ["example_id", "query", "product_locale", "esci_label", "product_title", "product_description", "product_bullet_point",
            "product_brand", "product_color"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    df = df[df.product_locale == "us"]
    by_label = {lab: g.sample(n=min(len(g), MAX_Q // 4), random_state=r.randrange(2**32)) for lab, g in df.groupby("esci_label")}
    items = []
    for lab, g in sorted(by_label.items()):
        for row in g.itertuples():
            parts = [f"Search query: {row.query.strip()}", f"Product title: {row.product_title}"]
            for k, v in [("Brand", row.product_brand), ("Colour", row.product_color),
                         ("Bullet points", row.product_bullet_point), ("Description", row.product_description)]:
                if isinstance(v, str) and v.strip() and v.strip() != "None":
                    parts.append(f"{k}: {v.strip()}")
            q = q_choice(r.choice(["How relevant is this product to the shopper's search?",
                                   "Does this product match what the customer searched for?",
                                   "How well does this product fit the search query?"]),
                         {k.lower(): v for k, v in ESCI.items()}, lab.lower())
            items.append((int(row.example_id), "\n".join(parts), {"relevance": q}))
    r.shuffle(items)
    return capped("esci", items)


CONFUSABLE = [{"Media", "Movies"}, {"Hotels", "Homes"}]  # domains whose descriptions overlap; never distractors for each other


def sgd(files):
    """One item per user turn that starts a new active intent: which intent (options from the dialogue's services), or which
    service (the true one against 3 services of other domains)."""
    r = random.Random("ext2:sgd")
    dlg, sch = (pd.read_parquet(f) for f in files)
    svc_desc = dict(zip(sch.service_name, sch.description))
    intents = {s: dict(zip(it["name"], it["description"])) for s, it in zip(sch.service_name, sch.intents)}
    key = lambda s, n: s.split("_")[0].lower() + "_" + re.sub(r"(?<!^)(?=[A-Z])", "_", n).lower()
    items = []
    for row in dlg.itertuples():
        t, hist, prev = row.turns, [], None
        for k, (spk, utt, fr) in enumerate(zip(t["speaker"], t["utterance"], t["frames"])):
            hist.append(f"{'User' if spk == 0 else 'Assistant'}: {utt}")
            if spk != 0:
                continue
            act = [(s, st["active_intent"]) for s, st in zip(fr["service"], fr["state"]) if st["active_intent"] not in ("", "NONE")]
            if len(act) != 1 or act[0] == prev:
                continue
            prev = act[0]
            svc, intent = act[0]
            if r.random() < 0.5:
                crit = {key(s, n): f"{n}: {d}" for s in row.services for n, d in intents[s].items()}
                n_opts = sum(len(intents[s]) for s in row.services)
                if n_opts > 10 or len(crit) != n_opts:
                    continue  # too many options, or two services of one domain share an intent name
                q = {"user_intent": q_choice(r.choice(["What is the user currently trying to get done?",
                                                       "Which task is the user working on at this point in the conversation?"]),
                                             crit, key(svc, intent))}
            else:
                dom = svc.split("_")[0]
                near = next((g for g in CONFUSABLE if dom in g), {dom})
                others = r.sample(sorted({s for s in svc_desc if s.split("_")[0] not in near}), 3)
                crit, lab = lettered(r, [svc_desc[s] for s in [svc] + others], 0)
                q = {"service": q_choice(r.choice(["Which service should handle the user's latest message?",
                                                   "Which system does the user's last request need to go to?"]), crit, lab)}
            items.append((f"{row.dialogue_id}:{k}", "\n".join(hist), q))
    r.shuffle(items)
    return capped("sgd", items)


BUILDERS = {"hellaswag": hellaswag, "swag": swag, "cosmos_qa": cosmos_qa, "openbookqa": openbookqa, "newyorker": newyorker,
            "when2call": when2call, "toolace": toolace, "glaive_fc": glaive_fc, "ragtruth": ragtruth, "esci": esci, "sgd": sgd}


def build(dl: Path) -> dict[str, tuple[list[dict], list[dict]]]:
    """name -> (train rows, dev rows)."""
    out = {}
    for name in SOURCES:
        files = download(dl, name)
        out[name] = reasoning(name, files) if name in REASONING else BUILDERS[name](files)
        print(f"{name}: {len(out[name][0])} train, {len(out[name][1])} dev", flush=True)
    return out
