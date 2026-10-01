from __future__ import annotations

import sys
from types import SimpleNamespace

from inference.engine import Engine, GroupRow, pack_requests
from inference.types import Options


def packing() -> list[str]:
    failures = []
    for sizes, capacity in (([3, 9, 2, 8, 1, 0, 5], 8), ([1] * 10, 4), ([], 8), ([17], 8)):
        groups = pack_requests(sizes, capacity)
        if sorted(q for g in groups for q in g) != [(r, i) for r, size in enumerate(sizes) for i in range(size)]:
            failures.append(f"pack_requests({sizes}, {capacity}) does not hold every question once: {groups}")
        if any(not g or len(g) > capacity for g in groups):
            failures.append(f"pack_requests({sizes}, {capacity}) made an empty group or one above capacity: {groups}")
        for r, size in enumerate(sizes):
            if 0 < size <= capacity and sum(any(request == r for request, _ in g) for g in groups) != 1:
                failures.append(f"pack_requests({sizes}, {capacity}) split request {r}, which fits in one group: {groups}")
    return failures


def batching() -> list[str]:
    groups: list[list[GroupRow]] = []

    def group(rows: list[GroupRow]) -> list[tuple[str, int]]:
        groups.append(rows)
        return [(row.record.name, row.question) for row in rows]

    engine = SimpleNamespace(R=4, group=group)
    jobs = [(SimpleNamespace(name=f"request {j}", questions=list(range(size))), Options(think=think))
            for j, (size, think) in enumerate(((3, True), (1, False), (5, True), (2, False), (0, True), (2, True)))]
    failures = []
    results = Engine.answer_batch(engine, jobs)
    if results != [[(record.name, i) for i in record.questions] for record, _ in jobs]:
        failures.append(f"answer_batch did not return each question's result in its place: {results}")
    if any(len(rows) > engine.R or len({row.options.think for row in rows}) > 1 for rows in groups):
        failures.append("answer_batch made a group above max_rows or mixed thinking and no-think rows")
    return failures


def main() -> None:
    failures = packing() + batching()
    print("\n".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
