from __future__ import annotations

import glob
import json
import random

import torch

from loader.dataloader import Encoder


def load_streams(chain_cache: str, encoder: Encoder, limit: int = 0, seed: int = 0) -> list[list[int]]:
    streams = []
    paths = sorted(p for pattern in chain_cache.split(",") for p in glob.glob(pattern))
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            chain = row["chain"]
            if encoder.think_end_id in chain:
                chain = chain[:chain.index(encoder.think_end_id) + 1]
            streams.append(row["prompt"] + chain)
    rng = random.Random(seed)
    rng.shuffle(streams)
    return streams[:limit] if limit else streams


def pack(streams: list[list[int]], length: int, sep: int) -> torch.Tensor:
    flat: list[int] = []
    for s in streams:
        flat.extend(s)
        flat.append(sep)
    n = len(flat) // length
    return torch.tensor(flat[: n * length], dtype=torch.long).view(n, length)


class Batches:
    def __init__(self, packed: torch.Tensor, batch_size: int, seed: int = 0):
        self.packed, self.batch_size = packed, batch_size
        self.gen = torch.Generator().manual_seed(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.packed.shape[0] // self.batch_size

    def __iter__(self):
        order = torch.randperm(self.packed.shape[0], generator=self.gen)
        for start in range(0, len(order) - self.batch_size + 1, self.batch_size):
            yield self.packed[order[start:start + self.batch_size]]
        self.epoch += 1
