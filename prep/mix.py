"""The training mix as ordered lists of row ids into a pool of rows built from public sources.

A few sources reuse an id for different rows; the pool keys the second such row `<id>@2`, the third `<id>@3`, in build order.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "dev", "rl")


def as_read(row: dict) -> dict:
    """The row as it reads back from a jsonl file."""
    return json.loads(json.dumps(row, ensure_ascii=False, default=str))


def canonical(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


class Pool:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.canon: dict[str, str] = {}
        self.ascii: set[str] = set()  # rows the original mixes wrote with ASCII escapes

    def add(self, records, ascii: bool = False) -> None:
        for r in records:
            r = as_read(r)
            c, base = canonical(r), r["_meta"]["id"]
            key, n = base, 1
            while key in self.rows and self.canon[key] != c:
                n += 1
                key = f"{base}@{n}"
            self.rows[key], self.canon[key] = r, c
            if ascii:
                self.ascii.add(key)

    def key(self, row: dict) -> str | None:
        """The pool key of a row read from a file, or None if the pool has no equal row."""
        c, base = canonical(row), row["_meta"]["id"]
        key, n = base, 1
        while key in self.rows:
            if self.canon[key] == c:
                return key
            n += 1
            key = f"{base}@{n}"
        return None

    def line(self, key: str) -> str:
        return json.dumps(self.rows[key], ensure_ascii=key in self.ascii) + "\n"


def read_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return fh.read().splitlines()


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:  # mtime 0: the same ids give the same bytes
        fh.write("".join(i + "\n" for i in ids).encode())


def write_split(pool: Pool, ids: list[str], path: Path) -> dict:
    missing = [i for i in ids if i not in pool.rows]
    if missing:
        raise KeyError(f"{path.name}: {len(missing)} ids are not in the pool, e.g. {missing[:5]}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i in ids:
            fh.write(pool.line(i))
    rows = [pool.rows[i] for i in ids]
    return {"records": len(rows), "questions": sum(len(r["questions"]) for r in rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sources": dict(sorted(Counter(r["_meta"]["source"] for r in rows).items()))}


def ids_of(path: Path, pool: Pool) -> tuple[list[str], list[dict]]:
    """Pool keys of the rows of a mix file, in file order, and the rows the pool does not have."""
    ids, unmatched = [], []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        key = pool.key(row)
        if key is None:
            unmatched.append(row)
        else:
            ids.append(key)
    return ids, unmatched
