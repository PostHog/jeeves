from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable

from datasets import load_dataset

PINNED_REVISIONS = {
    "CogComp/trec": "65752bf53af25bc935a0dce92fb5b6c930728450",
    "legacy-datasets/banking77": "f54121560de48f2852f90be299010d1d6dc612ec",
    "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "fancyzhx/ag_news": "eb185aade064a813bc0b7f42de02595523103ca4",
    "SetFit/amazon_reviews_multi_en": "ec73b665e4be0f567b69d39425355401cfe0d29b",
    "fancyzhx/dbpedia_14": "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "SetFit/sst5": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
    "nyu-mll/multi_nli": "da70db2af9d09693783c3320c4249840212ee221",
    "Yelp/yelp_review_full": "c1f9ee939b7d05667af864ee1cb066393154bf85",
}
PARQUET_BRANCH = {"CogComp/trec": "refs/convert/parquet", "legacy-datasets/banking77": "refs/convert/parquet"}

AG = {"world": "World news: politics, international affairs, conflicts", "sports": "Sports: games, athletes, teams, results",
      "business": "Business: companies, markets, economy, finance",
      "scitech": "Science and technology: research, gadgets, software, space"}
MNLI = {"entailment": "The hypothesis follows from the premise",
        "neutral": "The hypothesis may or may not be true given the premise",
        "contradiction": "The hypothesis contradicts the premise"}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
YELP = ["1 star: terrible experience", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
AMAZON = ["1 star: very negative", "2 stars: negative", "3 stars: mixed", "4 stars: positive", "5 stars: very positive"]
TREC = {"abbreviation": "Asks what an abbreviation stands for",
        "entity": "Asks about a thing, object, animal, product, or creative work",
        "description": "Asks for a definition, description, reason, or manner",
        "human": "Asks about a person, group, or organisation", "location": "Asks about a place",
        "number": "Asks for a number, date, count, or other numeric value"}
EMOTION = {"sadness": None, "joy": None, "love": None, "anger": None, "fear": None, "surprise": None}
BANK_TEMPLATES = ["Customer asks about {}", "Issue concerning {}", "Request related to {}", "{}"]
TEXT_FIELDS = ("text", "premise", "passage", "content", "question", "sentence")


def source_seed(seed: Any, source: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


class Source(random.Random):
    def __init__(self, seed: int, revision: str | None = None):
        super().__init__(seed)
        self.revision = revision
        self.origins: list[dict[str, Any]] = []


def load(repo: str, split: str, revision: str | None):
    name, _, config = repo.partition(":")
    candidates = [revision, PARQUET_BRANCH.get(name), None]
    errors = []
    for rev in dict.fromkeys(candidates):
        try:
            return load_dataset(name, config or None, split=split, revision=rev)
        except Exception as e:
            errors.append(f"{rev}: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError(f"could not load {repo} [{split}]: " + " | ".join(errors))


def wrap_state(text: str, rng: random.Random) -> Any:
    r = rng.random()
    if r < 0.15:
        return {"document": text}
    if r < 0.25:
        return {"ticket": {"channel": rng.choice(["email", "chat", "web form"]), "body": text}}
    if r < 0.32:
        return [{"role": "customer", "content": text}]
    return text


def instr(text: str, rng: random.Random) -> Any:
    if rng.random() < 0.15:
        return {"question": text, "focus": rng.choice(["Use only the information given.", "Pick the single best fit.",
                                                       "Consider the whole message."])}
    return text


def desc(text: str, rng: random.Random, p_null: float = 0.3, p_struct: float = 0.1) -> Any:
    r = rng.random()
    if r < p_null:
        return None
    if r < p_null + p_struct:
        return {"what": text}
    return text


def sample(ds, n: int, src: Source) -> list[dict[str, Any]]:
    rows = []
    src.origins = []
    for i in src.sample(range(len(ds)), min(n, len(ds))):
        row = ds[i]
        if row.get("label", 0) == -1:
            continue
        text = next((row[k] for k in TEXT_FIELDS if isinstance(row.get(k), str)), json.dumps(row, sort_keys=True))
        normalized = " ".join(text.casefold().split())
        src.origins.append({"row": i, "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                            "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()})
        rows.append(row)
    return rows


def question(qtype: str, instructions: Any, label: Any, src: str, criteria: Any = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": qtype, "instructions": instructions}
    if criteria is not None or qtype != "noul":
        q["criteria"] = criteria
    q["label"] = label
    q["src"] = src
    return q


def banking77(split, n, rng):
    ds = load("legacy-datasets/banking77", split, rng.revision)
    names = ds.features["label"].names
    out = []
    for ex in sample(ds, n, rng):
        t = rng.choice(BANK_TEMPLATES)
        crit = {k: desc(t.format(k.replace("_", " ")), rng, p_null=0.5, p_struct=0.0) for k in names}
        q = question("choice", instr("Which banking intent best describes this customer message?", rng),
                     names[ex["label"]], "banking77", crit)
        out.append({"state": wrap_state(ex["text"], rng), "questions": {"intent": q}})
    return out


def boolq(split, n, rng):
    ds = load("google/boolq", split, rng.revision)
    out = []
    for ex in sample(ds, n, rng):
        crit = None
        if rng.random() < 0.4:
            crit = {"true": "The passage supports a yes answer", "false": "The passage supports a no answer or does not say"}
        q = question("noul", instr(ex["question"].strip().rstrip("?") + "?", rng), bool(ex["answer"]), "boolq", crit)
        out.append({"state": wrap_state(ex["passage"], rng), "questions": {"answer": q}})
    return out


def agnews(split, n, rng):
    ds = load("fancyzhx/ag_news", split, rng.revision)
    keys = list(AG)
    out = []
    for ex in sample(ds, n, rng):
        y = keys[ex["label"]]
        qs = {"topic": question("choice", instr("What is the topic of this article?", rng), y, "agnews",
                                {k: desc(v, rng) for k, v in AG.items()})}
        for k in rng.sample(keys, 2):
            qs[f"is_{k}"] = question("noul", f"Is this article about {AG[k].split(':')[0].lower()}?", k == y, "agnews_yn")
        out.append({"state": wrap_state(ex["text"], rng), "questions": qs})
    return out


def mnli(split, n, rng):
    ds = load("nyu-mll/multi_nli", split, rng.revision)
    keys = list(MNLI)
    out = []
    for ex in sample(ds, n, rng):
        if ex["label"] < 0:
            continue
        q = question("choice", instr(f'Hypothesis: "{ex["hypothesis"]}" How does it relate to the premise?', rng),
                     keys[ex["label"]], "mnli", {k: desc(v, rng) for k, v in MNLI.items()})
        out.append({"state": wrap_state(ex["premise"], rng), "questions": {"relation": q}})
    return out


def sst5(split, n, rng):
    ds = load("SetFit/sst5", split, rng.revision)
    return [{"state": wrap_state(ex["text"], rng),
             "questions": {"sentiment": question("score", instr("What is the sentiment of this review sentence?", rng),
                                                 ex["label"], "sst5", list(SST5))}} for ex in sample(ds, n, rng)]


def yelp(split, n, rng):
    ds = load("Yelp/yelp_review_full", split, rng.revision)
    out = []
    for ex in sample(ds, n, rng):
        text = " ".join(ex["text"].split()[:220])
        qs = {"rating": question("score", instr("How many stars did this reviewer give?", rng), ex["label"], "yelp", list(YELP)),
              "recommend": question("noul", "Would this reviewer recommend the business?", ex["label"] >= 3, "yelp_yn",
                                    {"true": "Clearly positive overall", "false": "Negative or mixed"})}
        out.append({"state": wrap_state(text, rng), "questions": qs})
    return out


def trec(split, n, rng):
    ds = load("CogComp/trec", split, rng.revision)
    keys = list(TREC)
    return [{"state": wrap_state(ex["text"], rng),
             "questions": {"answer_type": question("choice", "What kind of answer does this question ask for?",
                                                   keys[ex["coarse_label"]], "trec", dict(TREC))}} for ex in sample(ds, n, rng)]


def dbpedia14(split, n, rng):
    ds = load("fancyzhx/dbpedia_14", split, rng.revision)
    names = [x.lower().replace(" ", "_") for x in ds.features["label"].names]
    return [{"state": wrap_state(" ".join(ex["content"].split()[:200]), rng),
             "questions": {"category": question("choice", "Which category does the subject of this encyclopedia text belong to?",
                                                names[ex["label"]], "dbpedia14", {k: None for k in names})}}
            for ex in sample(ds, n, rng)]


def emotion(split, n, rng):
    ds = load("dair-ai/emotion:split", split, rng.revision)
    keys = list(EMOTION)
    return [{"state": ex["text"],
             "questions": {"emotion": question("choice", "Which emotion does the writer express?", keys[ex["label"]], "emotion",
                                               dict(EMOTION))}} for ex in sample(ds, n, rng)]


def imdb(split, n, rng):
    ds = load("stanfordnlp/imdb", split, rng.revision)
    return [{"state": wrap_state(" ".join(ex["text"].replace("<br />", " ").split()[:220]), rng),
             "questions": {"positive": question("noul", "Is this movie review positive?", ex["label"] == 1, "imdb",
                                                {"true": "The reviewer liked the film overall",
                                                 "false": "The reviewer disliked the film overall"})}}
            for ex in sample(ds, n, rng)]


def amazon(split, n, rng):
    ds = load("SetFit/amazon_reviews_multi_en", split, rng.revision)
    return [{"state": wrap_state(" ".join(ex["text"].split()[:220]), rng),
             "questions": {"stars": question("score", "How many stars did this product reviewer give?", ex["label"], "amazon",
                                             list(AMAZON))}} for ex in sample(ds, n, rng)]


def qnli(split, n, rng):
    ds = load("nyu-mll/glue:qnli", split, rng.revision)
    return [{"state": wrap_state(ex["sentence"], rng),
             "questions": {"answers": question("noul", f'Does the sentence contain the answer to this question: "{ex["question"]}"',
                                               ex["label"] == 0, "qnli")}} for ex in sample(ds, n, rng)]


def tweet_offensive(split, n, rng):
    ds = load("cardiffnlp/tweet_eval:offensive", split, rng.revision)
    return [{"state": ex["text"],
             "questions": {"offensive": question("noul", "Is this post offensive?", ex["label"] == 1, "tweet_offensive",
                                                 {"true": "Contains insults, threats, profanity directed at someone, or hateful content",
                                                  "false": "Not offensive"})}} for ex in sample(ds, n, rng)]


def mmlu(split, n, rng):
    ds = load("cais/mmlu:all", split, rng.revision)
    keys = ["a", "b", "c", "d"]
    return [{"state": {"subject": ex["subject"].replace("_", " "), "question": ex["question"]},
             "questions": {"answer": question("choice", "Which option correctly answers the question?", keys[ex["answer"]], "mmlu",
                                              dict(zip(keys, ex["choices"])))}} for ex in sample(ds, n, rng)]


def paws(split, n, rng):
    ds = load("google-research-datasets/paws:labeled_final", split, rng.revision)
    return [{"state": wrap_state(ex["sentence1"], rng),
             "questions": {"paraphrase": question("noul", f'Does this sentence mean the same thing: "{ex["sentence2"]}"',
                                                  ex["label"] == 1, "paws",
                                                  {"true": "Same meaning, possibly reworded",
                                                   "false": "Different meaning, even if most words match"})}}
            for ex in sample(ds, n, rng)]


def sciq(split, n, rng):
    ds = load("allenai/sciq", split, rng.revision)
    out = []
    for ex in sample(ds, n, rng):
        options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        keys = ["a", "b", "c", "d"]
        order = list(range(4))
        rng.shuffle(order)
        crit = {keys[i]: options[j] for i, j in enumerate(order)}
        state = {"passage": ex["support"], "question": ex["question"]} if ex["support"] else {"question": ex["question"]}
        out.append({"state": state, "questions": {"answer": question("choice", "Which option answers the science question?",
                                                                     keys[order.index(0)], "sciq", crit)}})
    return out


Converter = Callable[[str, int, Source], list[dict[str, Any]]]

SOURCES: dict[str, tuple[Converter, str, str]] = {
    "banking77": (banking77, "train", "test"), "boolq": (boolq, "train", "validation"), "agnews": (agnews, "train", "test"),
    "mnli": (mnli, "train", "validation_matched"), "sst5": (sst5, "train", "test"), "yelp": (yelp, "train", "test"),
    "trec": (trec, "train", "test"), "dbpedia14": (dbpedia14, "train", "test"), "amazon": (amazon, "train", "test"),
    "imdb": (imdb, "train", "test"),
    "emotion": (emotion, "train", "test"), "qnli": (qnli, "train", "validation"), "tweet_offensive": (tweet_offensive, "train", "test"),
    "mmlu": (mmlu, "test", "test"), "paws": (paws, "train", "test"), "sciq": (sciq, "train", "test"),
}
REPOS = {
    "banking77": "legacy-datasets/banking77", "boolq": "google/boolq", "agnews": "fancyzhx/ag_news", "mnli": "nyu-mll/multi_nli",
    "sst5": "SetFit/sst5", "yelp": "Yelp/yelp_review_full", "trec": "CogComp/trec", "dbpedia14": "fancyzhx/dbpedia_14",
    "amazon": "SetFit/amazon_reviews_multi_en", "imdb": "stanfordnlp/imdb", "emotion": "dair-ai/emotion:split",
    "qnli": "nyu-mll/glue:qnli", "tweet_offensive": "cardiffnlp/tweet_eval:offensive", "mmlu": "cais/mmlu:all",
    "paws": "google-research-datasets/paws:labeled_final", "sciq": "allenai/sciq",
}
TRAINABLE = ("banking77", "boolq", "agnews", "mnli", "sst5", "yelp", "trec", "dbpedia14", "amazon", "imdb")
EVAL_ONLY = ("mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq")


def build(source: str, split: str, n: int, seed: Any, revision: str | None) -> list[dict[str, Any]]:
    fn, train_split, eval_split = SOURCES[source]
    source_split = train_split if split == "train" else eval_split
    src = Source(source_seed(seed, source), revision)
    records = fn(source_split, n, src)
    if len(records) != len(src.origins):
        raise ValueError(f"provenance mismatch for {source}")
    for record, origin in zip(records, src.origins):
        record["_meta"] = {**origin, "source": source, "repo": REPOS[source].partition(":")[0], "revision": revision,
                           "split": source_split, "id": f"{source}/{source_split}/{origin['row']}"}
    return records
