from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

CHOICE, NOUL, SCORE = "choice", "noul", "score"
QUESTION_TYPES = (CHOICE, NOUL, SCORE)


def render(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in value)
    return "\n".join(
        f"{pad}{k}:\n{render(v, indent + 1)}" if isinstance(v, (dict, list)) else f"{pad}{k}: {render(v)}"
        for k, v in value.items()
    )


def option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def state_hash(state: Any) -> str:
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


@dataclass
class Question:
    id: str
    type: str
    instructions: Any
    criteria: Any = None
    label: Any = None
    src: str = ""
    target: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise ValueError(f"unknown question type {self.type!r}")
        if self.type == CHOICE and not isinstance(self.criteria, dict):
            raise ValueError(f"{self.id}: choice criteria must be a dict")
        if self.type == SCORE and not isinstance(self.criteria, list):
            raise ValueError(f"{self.id}: score criteria must be a list")
        if self.type == NOUL and self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError(f"{self.id}: noul criteria must be a dict or null")

    def keys(self) -> list[str]:
        if self.type == CHOICE:
            return list(self.criteria)
        if self.type == NOUL:
            return ["false", "true"]
        return [str(i) for i in range(len(self.criteria))]

    def instruction_text(self) -> str:
        return render(self.instructions)

    def options(self) -> list[str]:
        if self.type == CHOICE:
            return [option_text(k, v) for k, v in self.criteria.items()]
        if self.type == NOUL:
            c = self.criteria or {}
            return [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        return [render(x) for x in self.criteria]

    def label_index(self) -> int:
        if self.type == CHOICE:
            return self.keys().index(self.label)
        return int(self.label)

    def target_vector(self) -> list[float] | None:
        if self.target is None:
            return None
        t = [float(self.target.get(k, 0.0)) for k in self.keys()]
        s = sum(t)
        if s <= 0:
            raise ValueError(f"{self.id}: target puts no mass on any option")
        return [x / s for x in t]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None or self.type != NOUL:
            d["criteria"] = self.criteria
        d["label"] = self.label
        d["src"] = self.src
        if self.target is not None:
            d["target"] = self.target
        return d

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> "Question":
        return cls(id=qid, type=d["type"], instructions=d["instructions"], criteria=d.get("criteria"),
                   label=d.get("label"), src=d.get("src", ""), target=d.get("target"))


@dataclass
class DataFormat:
    state: Any
    questions: list[Question]
    meta: dict[str, Any] = field(default_factory=dict)

    def state_text(self) -> str:
        return render(self.state)

    def question(self, qid: str) -> Question:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(qid)

    @property
    def source(self) -> str:
        return self.meta.get("source", "")

    @property
    def id(self) -> str:
        return self.meta.get("id", "")

    def text_sha256(self) -> str:
        return self.meta.get("text_sha256") or state_hash(self.state)

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "questions": {q.id: q.to_dict() for q in self.questions}, "_meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataFormat":
        if "state" not in d or not isinstance(d.get("questions"), dict) or not d["questions"]:
            raise ValueError("a record needs a state and a non-empty questions object")
        return cls(state=d["state"], questions=[Question.from_dict(k, v) for k, v in d["questions"].items()],
                   meta=dict(d.get("_meta", {})))

    @classmethod
    def from_json(cls, line: str) -> "DataFormat":
        return cls.from_dict(json.loads(line))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


def read_jsonl(path: str | os.PathLike) -> list[DataFormat]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    out.append(DataFormat.from_json(line))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"{os.fspath(path)}:{n}: {e}") from e
    return out


def write_jsonl(path: str | os.PathLike, records: Iterable[DataFormat | dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write((r.to_json() if isinstance(r, DataFormat) else json.dumps(r, ensure_ascii=False, default=str)) + "\n")
            n += 1
    return n


def by_source(records: Sequence[DataFormat]) -> dict[str, list[DataFormat]]:
    out: dict[str, list[DataFormat]] = {}
    for r in records:
        out.setdefault(r.source, []).append(r)
    return out
