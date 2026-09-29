from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from prep.format import DataFormat, Question, read_jsonl

STATE, Q, OPT, OPT_END, DECIDE = "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"
THINK, THINK_END = "<think>", "</think>"
CONTROL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def sanitize(text: str) -> str:
    return CONTROL_RE.sub(r"<¦\1¦>", text)


@dataclass
class Example:
    prompt: list[int]
    remainder: list[int]
    label: int
    target: list[float] | None
    n_options: int
    source: str
    source_id: int
    record_id: str
    question_id: str
    think: list[int] | None = None
    plain_prompt: list[int] | None = None

    @property
    def full(self) -> list[int]:
        return self.prompt + self.remainder


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    opt_index: torch.Tensor
    opt_mask: torch.Tensor
    decide_index: torch.Tensor
    labels: torch.Tensor
    targets: torch.Tensor
    source_ids: torch.Tensor
    chain_mask: torch.Tensor
    valid: torch.Tensor
    group_ids: torch.Tensor
    think_start: torch.Tensor
    suffix_len: torch.Tensor
    old_logp: torch.Tensor

    def to(self, device, non_blocking: bool = True) -> "Batch":
        return Batch(**{k: v.to(device, non_blocking=non_blocking) for k, v in self.__dict__.items()})

    def pin_memory(self) -> "Batch":
        return Batch(**{k: v.pin_memory() for k, v in self.__dict__.items()})

    def __len__(self) -> int:
        return self.input_ids.shape[0]


@dataclass(frozen=True)
class Markers:
    pad_id: int
    opt_end_id: int
    think_end_id: int
    empty_think: tuple[int, ...]
    pad_multiple: int


class Encoder:
    FORMAT = "markers-v3-plainchains"

    def __init__(self, tokenizer_repo: str = "Qwen/Qwen3.5-9B", pad_multiple: int = 128):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_repo)
        self.pad_multiple = pad_multiple
        ids = self.tok.convert_tokens_to_ids([STATE, Q, OPT, OPT_END, DECIDE, THINK, THINK_END])
        if any(i is None or i == self.tok.unk_token_id for i in ids):
            raise ValueError("marker tokens missing from tokenizer")
        self.state_id, self.q_id, self.opt_id, self.opt_end_id, self.decide_id, self.think_id, self.think_end_id = ids
        self.pad_id = self.tok.pad_token_id
        special = set(self.tok.all_special_ids) | set(self.tok.get_added_vocab().values())
        self.banned_ids = sorted(special - {self.think_end_id})
        self.empty_think = self.encode("\n")

    @property
    def markers(self) -> Markers:
        return Markers(self.pad_id, self.opt_end_id, self.think_end_id, tuple(self.empty_think), self.pad_multiple)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    @staticmethod
    def option_block(q: Question) -> str:
        return "".join(f"{OPT}{sanitize(o)}{OPT_END}\n" for o in q.options())

    def chat(self, content: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)

    def prompt_text(self, record: DataFormat, q: Question) -> str:
        return self.chat(f"{STATE}{sanitize(record.state_text())}\n{Q}{sanitize(q.instruction_text())}\n{self.option_block(q)}")

    def plain_prompt_text(self, record: DataFormat, q: Question) -> str:
        options = "\n".join(sanitize(o) for o in q.options())
        return self.chat(f"Context:\n{sanitize(record.state_text())}\n\nQuestion: {sanitize(q.instruction_text())}\n\nOptions:\n{options}")

    def remainder_text(self, q: Question) -> str:
        return f"{THINK_END}\n\n{self.option_block(q)}{DECIDE}"

    def example(self, record: DataFormat, q: Question, source_id: int) -> Example:
        prompt = self.encode(self.prompt_text(record, q))
        remainder = self.encode(self.remainder_text(q))
        if prompt[-2:] != self.encode(f"{THINK}\n") or remainder[0] != self.think_end_id or remainder[-1] != self.decide_id:
            raise ValueError(f"unexpected layout for {record.id}/{q.id}")
        if remainder.count(self.opt_end_id) != len(q.options()):
            raise ValueError(f"option boundary count mismatch for {record.id}/{q.id}")
        return Example(prompt=prompt, remainder=remainder, label=q.label_index(), target=q.target_vector(),
                       n_options=len(q.options()), source=record.source, source_id=source_id, record_id=record.id,
                       question_id=q.id, plain_prompt=self.encode(self.plain_prompt_text(record, q)))


def build_examples(records: Sequence[DataFormat], encoder: Encoder, sources: dict[str, int], max_len: int,
                   extra_len: int = 0) -> tuple[list[Example], int]:
    out, dropped = [], 0
    for r in records:
        sid = sources.setdefault(r.source, len(sources))
        for q in r.questions:
            ex = encoder.example(r, q, sid)
            if len(ex.prompt) + len(ex.remainder) + extra_len > max_len:
                dropped += 1
                continue
            out.append(ex)
    return out, dropped


def load_examples(path: str, encoder: Encoder, sources: dict[str, int], max_len: int, extra_len: int = 0,
                  limit: int = 0, seed: int = 0) -> tuple[list[Example], int]:
    records = read_jsonl(path)
    if limit and limit < len(records):
        records = random.Random(seed).sample(records, limit)
    return build_examples(records, encoder, sources, max_len, extra_len)


def readout_positions(ids: list[int], m: Markers) -> tuple[list[int], int]:
    start = len(ids) - 1 - ids[::-1].index(m.think_end_id)
    opts = [i for i in range(start, len(ids)) if ids[i] == m.opt_end_id]
    return opts, len(ids) - 1


def padded_length(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple if multiple > 1 else n


@dataclass
class CollateRow:
    example: Example
    input_ids: list[int]
    group_id: int
    closed: bool = True
    chain_span: tuple[int, int] | None = None
    logprobs: list[float] | None = None


def collate(rows: list[CollateRow], m: Markers) -> Batch:
    B = len(rows)
    T = padded_length(max(len(row.input_ids) for row in rows), m.pad_multiple)
    K = max(row.example.n_options for row in rows)
    input_ids = torch.full((B, T), m.pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.long)
    opt_index = torch.zeros(B, K, dtype=torch.long)
    opt_mask = torch.zeros(B, K, dtype=torch.bool)
    decide_index = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    targets = torch.zeros(B, K, dtype=torch.float32)
    source_ids = torch.zeros(B, dtype=torch.long)
    chain_mask = torch.zeros(B, T, dtype=torch.bool)
    valid = torch.zeros(B, dtype=torch.bool)
    group_ids = torch.zeros(B, dtype=torch.long)
    think_start = torch.zeros(B, dtype=torch.long)
    suffix_len = torch.zeros(B, dtype=torch.long)
    old_logp = torch.zeros(B, T, dtype=torch.float32)
    for b, row in enumerate(rows):
        ex, ids, chain = row.example, row.input_ids, row.chain_span
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[b, :n] = 1
        opts, decide = readout_positions(ids, m)
        if len(opts) != ex.n_options:
            raise ValueError(f"expected {ex.n_options} option boundaries, found {len(opts)} for {ex.record_id}")
        opt_index[b, :ex.n_options] = torch.tensor(opts, dtype=torch.long) + b * T
        opt_mask[b, :ex.n_options] = True
        decide_index[b] = decide + b * T
        labels[b] = ex.label
        if ex.target is not None:
            targets[b, :ex.n_options] = torch.tensor(ex.target, dtype=torch.float32)
        else:
            targets[b, ex.label] = 1.0
        source_ids[b] = ex.source_id
        if chain is not None:
            chain_mask[b, chain[0]:chain[1]] = True
        valid[b] = row.closed
        group_ids[b] = row.group_id
        think_start[b] = len(ex.prompt)
        suffix_len[b] = len(ex.remainder)
        if row.logprobs is not None and chain is not None:
            span = chain[1] - chain[0]
            old_logp[b, chain[0]:chain[1]] = torch.tensor(row.logprobs[:span], dtype=torch.float32)
    return Batch(input_ids, attention_mask, opt_index, opt_mask, decide_index, labels, targets, source_ids,
                 chain_mask, valid, group_ids, think_start, suffix_len, old_logp)


def collate_sft(examples: list[Example], m: Markers) -> Batch:
    rows = []
    for i, ex in enumerate(examples):
        if ex.think is None:
            rows.append(CollateRow(example=ex, input_ids=ex.prompt + list(m.empty_think) + ex.remainder, group_id=i))
        else:
            chain, ok = split_chain(ex.think, m)
            rows.append(CollateRow(example=ex, input_ids=ex.prompt + chain + ex.remainder, group_id=i, closed=ok))
    return collate(rows, m)


def collate_prompts(examples: list[Example], m: Markers, plain: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [ex.plain_prompt if plain else ex.prompt for ex in examples]
    B, T = len(examples), padded_length(max(len(p) for p in prompts), m.pad_multiple)
    input_ids = torch.full((B, T), m.pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.long)
    for b, p in enumerate(prompts):
        n = len(p)
        input_ids[b, T - n:] = torch.tensor(p, dtype=torch.long)
        attention_mask[b, T - n:] = 1
    return input_ids, attention_mask


def split_chain(generated: list[int], m: Markers) -> tuple[list[int], bool]:
    if m.think_end_id in generated:
        return generated[:generated.index(m.think_end_id)], True
    return generated, False


def collate_rollouts(examples: list[Example], generated: list[list[int]], group_ids: list[int], m: Markers,
                     logprobs: list[list[float] | None] | None = None) -> Batch:
    rows = []
    for i, (ex, gen, group) in enumerate(zip(examples, generated, group_ids)):
        chain, ok = split_chain(gen, m)
        ids = ex.prompt + chain + ex.remainder
        span = (len(ex.prompt), len(ex.prompt) + len(chain) + int(ok))
        rows.append(CollateRow(example=ex, input_ids=ids, group_id=group, closed=ok, chain_span=span,
                               logprobs=logprobs[i] if logprobs is not None else None))
    return collate(rows, m)


def make_loader(examples: list[Example], batch_size: int, m: Markers, shuffle: bool, rank: int = 0, world_size: int = 1,
                seed: int = 0, drop_last: bool = True, num_workers: int = 2) -> DataLoader:
    sampler = DistributedSampler(examples, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
                                 drop_last=drop_last) if world_size > 1 else None
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      collate_fn=partial(collate_sft, m=m), num_workers=num_workers, pin_memory=True, drop_last=drop_last,
                      persistent_workers=num_workers > 0, generator=torch.Generator().manual_seed(seed))
