from __future__ import annotations

import json
from typing import Any

from inference.types import Options, Result
from prep.format import CHOICE, NOUL, QUESTION_TYPES, DataFormat, Question

MAX_OPTIONS = 255
MODEL_ID = "jeeves-latest"


def parse_options(raw: Any, defaults: Options) -> Options:
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError("options must be an object")
    unknown = set(raw) - {"think", "max_think", "nothink_threshold", "return_reasoning"}
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    think = raw.get("think", defaults.think)
    max_think = raw.get("max_think", defaults.max_think)
    threshold = raw.get("nothink_threshold", defaults.nothink_threshold)
    reasoning = raw.get("return_reasoning", defaults.return_reasoning)
    if not isinstance(think, bool) or not isinstance(reasoning, bool):
        raise ValueError("think and return_reasoning must be booleans")
    if isinstance(max_think, bool) or not isinstance(max_think, int) or max_think < 0:
        raise ValueError("max_think must be a non-negative integer")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
        raise ValueError("nothink_threshold must be null or a number in [0, 1]")
    return Options(think=think, max_think=max_think, nothink_threshold=None if threshold is None else float(threshold),
                   return_reasoning=reasoning)


def parse_question(qid: str, raw: Any) -> Question:
    if not isinstance(raw, dict):
        raise ValueError(f"question {qid!r} must be an object")
    kind = raw.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {qid!r}: type must be one of {list(QUESTION_TYPES)}")
    unknown = set(raw) - {"type", "instructions", "criteria"}
    if unknown:
        raise ValueError(f"question {qid!r}: unknown fields {sorted(unknown)}")
    criteria = raw.get("criteria")
    if kind == CHOICE and not (isinstance(criteria, dict) and 1 <= len(criteria) <= MAX_OPTIONS):
        raise ValueError(f"question {qid!r}: choice criteria must be an object with 1..{MAX_OPTIONS} options")
    if kind == NOUL and criteria is not None and not (isinstance(criteria, dict) and set(criteria) <= {"true", "false"}):
        raise ValueError(f"question {qid!r}: noul criteria may only describe true and false")
    if kind not in (CHOICE, NOUL) and not (isinstance(criteria, list) and 1 <= len(criteria) <= MAX_OPTIONS):
        raise ValueError(f"question {qid!r}: score criteria must be a list of 1..{MAX_OPTIONS} levels")
    return Question(id=qid, type=kind, instructions=raw.get("instructions"), criteria=criteria, label=0)


def parse_request(body: Any, defaults: Options) -> tuple[DataFormat, str, Options]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in body:
        raise ValueError("state is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    model = body.get("model", MODEL_ID)
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    record = DataFormat(state=body["state"], questions=[parse_question(str(k), v) for k, v in questions.items()])
    return record, model, parse_options(body.get("options"), defaults)


def r2(x: float) -> float:
    return round(float(x), 2)


def choice_confidence(p: list[float]) -> float:
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def score_confidence(p: list[float]) -> float:
    if len(p) == 1:
        return 1.0
    mode = max(range(len(p)), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (len(p) - 1)


def answer(q: Question, p: list[float]) -> dict:
    if q.type == NOUL:
        return {"type": NOUL, "noul": r2(p[1])}
    if q.type == CHOICE:
        keys = q.keys()
        return {"type": CHOICE, "choice": keys[max(range(len(p)), key=lambda i: p[i])], "confidence": r2(choice_confidence(p)),
                "probabilities": {k: r2(v) for k, v in zip(keys, p)}}
    return {"type": "score", "score": r2(sum(i * pi for i, pi in enumerate(p))), "legend": dict(zip(q.keys(), q.options())),
            "probabilities": {str(i): r2(v) for i, v in enumerate(p)}, "confidence": r2(score_confidence(p))}


def response(record: DataFormat, model: str, opts: Options, results: list[Result], tok, latency_ms: float) -> dict:
    answers = {q.id: answer(q, r.probs) for q, r in zip(record.questions, results)}
    out = {"model": model, "answers": answers,
           "usage": {"input_tokens": sum(r.prompt_tokens for r in results),
                     "output_tokens": len(tok(json.dumps(answers), add_special_tokens=False).input_ids),
                     "reasoning_tokens": sum(len(r.chain) for r in results)},
           "latency_ms": round(latency_ms, 1)}
    if opts.return_reasoning:
        out["reasoning"] = {q.id: {"thought": r.thought, "closed": r.closed, "tokens": len(r.chain),
                                   "text": tok.decode([t for t in r.chain if t != tok.convert_tokens_to_ids("</think>")])}
                            for q, r in zip(record.questions, results)}
    return out
