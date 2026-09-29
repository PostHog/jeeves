from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime
from typing import Any

from .contrastive import DATE_FAMILIES, FAMILIES, HELD_OUT_FAMILIES, UNDETERMINED, label_of, to_request, valid_pairs
from .format import render

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")
ROLE = {"purchase": "purchase", "request": "return request", "claim": "warranty claim", "promised": "promised delivery",
        "delivered": "delivery"}
STATEMENTS = {"Would this reviewer recommend the business?": "This reviewer recommends the business.",
              "Is this movie review positive?": "This movie review is positive.",
              "Is this article about science and technology?": "This article is about science and technology.",
              "Is this article about world news?": "This article is about world news.",
              "Is this article about sports?": "This article is about sports.",
              "Is this article about business?": "This article is about business."}
ASSERTION_TEMPLATES = ["This text is about {x}.", "The right category for this is {x}.", "This should be filed under {x}.",
                       "The topic here is {x}."]
ASSERTION_SOURCES = ("agnews", "dbpedia14", "trec", "banking77", "yelp", "imdb", "boolq")
UNKNOWABLE_FAMILIES = tuple(f for f in FAMILIES if f not in HELD_OUT_FAMILIES)


def date_facts(text: str) -> str:
    found: list[tuple[str, datetime]] = []
    for m in DATE_RE.finditer(text):
        raw = m.group(0)
        try:
            d = datetime.strptime(raw, "%B %d, %Y") if "," in raw else datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            continue
        if raw not in [r for r, _ in found]:
            found.append((raw, d))
    facts = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            n = (found[j][1] - found[i][1]).days
            if n == 0:
                facts.append(f"{found[j][0]} is the same day as {found[i][0]}.")
            else:
                facts.append(f"{found[j][0]} is {abs(n)} day{'s' if abs(n) != 1 else ''} {'after' if n > 0 else 'before'} {found[i][0]}.")
    return " ".join(facts)


def relational_sentence(it: dict) -> str | None:
    facts: dict = {}
    for _, f in it["sentences"]:
        facts.update(f)
    dates = [(k, v) for k, v in facts.items() if hasattr(v, "toordinal")]
    if len(dates) != 2:
        return None
    (ka, a), (kb, b) = sorted(dates, key=lambda kv: kv[1])
    n = (b - a).days
    return f"The {ROLE.get(kb, kb)} date was {n} day{'s' if n != 1 else ''} after the {ROLE.get(ka, ka)} date."


def dates_records(pairs_per_family: int, seed: Any) -> list[dict[str, Any]]:
    out = []
    for family in DATE_FAMILIES:
        for pair_id, a, b, order_seed in valid_pairs(family, pairs_per_family, f"{seed}:dates"):
            style = random.Random(f"{order_seed}:style").choice(["plain", "relational", "facts"])
            for sibling, it in (("a", a), ("b", b)):
                if style == "relational":
                    sent = relational_sentence(it)
                    it = {**it, "sentences": it["sentences"] + [(sent, {})]} if sent else it
                req = to_request(it, family, pair_id, sibling, random.Random(order_seed))
                if style == "facts":
                    facts = date_facts(render(req["state"]))
                    if facts:
                        req["state"] = {**req["state"], "date_facts": facts}
                req["_meta"].update(source="night2_dates", rendering=style, id=f"night2_dates/{family}/{pair_id}/{sibling}",
                                    group_id=f"night2_dates/{family}/{pair_id}")
                out.append(req)
    return out


def unknowable_records(pairs_per_family: int, seed: Any, families=UNKNOWABLE_FAMILIES) -> list[dict[str, Any]]:
    out = []
    for family in (families if families is not None else tuple(FAMILIES)):
        for pair_id, a, b, order_seed in valid_pairs(family, pairs_per_family, f"{seed}:unknowable"):
            for sibling, it in (("a", a), ("b", b)):
                control = to_request(it, family, pair_id, sibling, random.Random(order_seed))
                drop = next(i for i, (_, facts) in enumerate(it["sentences"]) if facts and label_of(it, drop=i) == UNDETERMINED)
                stripped = {**it, "sentences": [s for i, s in enumerate(it["sentences"]) if i != drop]}
                unk = to_request(stripped, family, pair_id, sibling, random.Random(order_seed))
                q = unk["questions"]["decision"]
                q["label"] = control["questions"]["decision"]["label"]
                q["src"] = f"unknowable_{family}"
                keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"] if q["type"] == "noul" \
                    else [str(i) for i in range(len(q["criteria"]))]
                q["target"] = {k: 1.0 / len(keys) for k in keys}
                control["questions"]["decision"]["src"] = f"unknowable_control_{family}"
                unk["_meta"].update(source="night2_unknowable", dropped_sentence=it["sentences"][drop][0],
                                    intact_label=control["questions"]["decision"]["label"],
                                    id=f"night2_unknowable/{family}/{pair_id}/{sibling}", group_id=f"night2_unknowable/{family}/{pair_id}",
                                    control_id=f"night2_unknowable_control/{family}/{pair_id}/{sibling}")
                control["_meta"].update(source="night2_unknowable_control", id=f"night2_unknowable_control/{family}/{pair_id}/{sibling}",
                                        group_id=f"night2_unknowable_control/{family}/{pair_id}")
                for r in (unk, control):
                    r["_meta"].pop("pair_id", None)
                    r["_meta"].pop("sibling", None)
                    body = {k: v for k, v in r.items() if k != "_meta"}
                    r["_meta"]["row_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                out += [unk, control]
    return out


def assertion_records(pool: list[dict[str, Any]], n: int, seed: Any) -> list[dict[str, Any]]:
    rows = [r for r in pool if r["_meta"]["source"] in ASSERTION_SOURCES]
    rng = random.Random(f"{seed}:assertion")
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n:
            break
        qs = {}
        for qid, q in r["questions"].items():
            if q["type"] == "choice" and len(q["criteria"]) >= 3:
                truth = q["label"]
                key = truth if rng.random() < 0.5 else rng.choice([k for k in q["criteria"] if k != truth])
                text = q["criteria"].get(key) or key.replace("_", " ")
                if not isinstance(text, str):
                    continue
                qs[f"{qid}_is"] = {"type": "noul", "instructions": rng.choice(ASSERTION_TEMPLATES).format(x=text.rstrip("?. ")),
                                   "label": key == truth, "src": f"night2_assertion_{r['_meta']['source']}"}
            elif q["type"] == "noul" and isinstance(q["instructions"], str) and q["instructions"] in STATEMENTS:
                qs[f"{qid}_stmt"] = {"type": "noul", "instructions": STATEMENTS[q["instructions"]], "label": q["label"],
                                     "src": f"night2_assertion_{r['_meta']['source']}"}
        if qs:
            out.append({"state": r["state"], "questions": qs,
                        "_meta": {**r["_meta"], "source": "night2_assertion", "id": "night2_assertion/" + r["_meta"]["id"],
                                  "group_id": r["_meta"]["group_id"], "parent_id": r["_meta"]["id"]}})
    return out
