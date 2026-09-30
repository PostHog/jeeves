from __future__ import annotations

from dataclasses import dataclass

PRECISIONS = ("bf16", "fp8")


@dataclass
class Options:
    think: bool = True
    max_think: int = 2560
    nothink_threshold: float | None = None
    return_reasoning: bool = False


@dataclass
class Result:
    probs: list[float]
    thought: bool
    chain: list[int]
    closed: bool
    nothink_probs: list[float] | None
    prompt_tokens: int

