from __future__ import annotations

import hashlib
import json
import random
from datetime import date, timedelta
from typing import Any

SHAPES = {
    "atom": 0,
    "negation": ("not", 0),
    "conjunction": ("and", 0, 1),
    "disjunction": ("or", 0, 1),
    "exception": ("unless", 0, 1),
    "conditional": ("if", 0, 1, 2),
    "nested_and": ("and", ("and", 0, 1), 2),
    "nested_or": ("or", 0, ("or", 1, 2)),
    "held_and_or": ("and", ("or", 0, 1), 2),
    "held_or_not": ("or", ("and", 0, 1), ("not", 2)),
    "held_conditional": ("if", 0, ("not", 1), 2),
    "final_combination": ("and", ("if", 0, 1, 2), 3),
    "final_negation": ("not", ("or", ("and", 0, 1), 2)),
    "final_exception": ("or", ("unless", 0, 1), 2),
}
TRAIN_SHAPES = tuple(list(SHAPES)[:8])
DEV_SHAPES = tuple(list(SHAPES)[8:11])
TEST_SHAPES = tuple(list(SHAPES)[11:])
KINDS = ("lt", "le", "gt", "ge", "eq", "range", "match", "elapsed", "flag")
NOUNS = ["request", "account", "package", "review", "member", "shipment", "entry", "case"]
PEOPLE = ["Mira", "Noah", "Aiko", "Ravi", "Sana", "Elin", "Tomas", "Kofi"]
POLICY_WRAPPERS = {
    0: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    1: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    2: "Approval requires the following rule to be true: {rule}. A false rule means denial. Routing references are irrelevant.",
    3: "A case is approved when {rule}; any other case is denied. Routing references play no part in the decision.",
    4: "DECISION RULE {rule} -> approve; otherwise reject. Ignore the routing reference.",
}
COMPARISONS = {"lt": "less than", "le": "at most", "gt": "greater than", "ge": "at least", "eq": "equal to"}


def atom_value(atom: dict, facts: dict) -> bool | None:
    if any(k not in facts for k in atom["fields"]):
        return None
    values = [facts[k] for k in atom["fields"]]
    kind, t, x = atom["kind"], atom["threshold"], values[0]
    if kind == "lt":
        return x < t
    if kind == "le":
        return x <= t
    if kind == "gt":
        return x > t
    if kind == "ge":
        return x >= t
    if kind == "eq":
        return x == t
    if kind == "range":
        return t <= x <= t + 10
    if kind == "match":
        return x == values[1]
    if kind == "elapsed":
        return (date.fromisoformat(values[1]) - date.fromisoformat(x)).days <= t
    if kind == "flag":
        return bool(x)
    raise ValueError(f"unknown atom kind {kind}")


def evaluate_rule(tree, atoms: list[dict], facts: dict) -> bool | None:
    if isinstance(tree, int):
        return atom_value(atoms[tree], facts)
    op, *children = tree
    values = [evaluate_rule(c, atoms, facts) for c in children]
    if op == "not":
        return None if values[0] is None else not values[0]
    if op == "unless":
        a, exception = values
        values = [a, None if exception is None else not exception]
        op = "and"
    if op == "and":
        return False if False in values else None if None in values else True
    if op == "or":
        return True if True in values else None if None in values else False
    if op == "if":
        condition, yes, no = values
        return yes if condition is True else no if condition is False else yes if yes == no else None
    raise ValueError(f"unknown operation {op}")


def atom_text(atom: dict) -> str:
    kind, fields, t = atom["kind"], atom["fields"], atom["threshold"]
    key = fields[0]
    if kind in COMPARISONS:
        return f"{key} is {COMPARISONS[kind]} {t}"
    if kind == "range":
        return f"{key} is between {t} and {t + 10}, including both endpoints"
    if kind == "match":
        return f"{fields[0]} is the same person as {fields[1]}"
    if kind == "elapsed":
        return f"the elapsed days from {fields[0]} to {fields[1]} are at most {t}"
    return f"{key} is yes"


def render_rule(tree, atoms: list[dict], style: int) -> str:
    if isinstance(tree, int):
        return atom_text(atoms[tree])
    op, *children = tree
    p = [render_rule(c, atoms, style) for c in children]
    if style == 3:
        if op == "not":
            return f"the condition \"{p[0]}\" fails"
        if op == "and":
            return f"{p[0]}, and also {p[1]}"
        if op == "or":
            return f"either {p[0]}, or else {p[1]}"
        if op == "unless":
            return f"{p[0]}, except when {p[1]}"
        return f"when {p[0]} the requirement is that {p[1]}, and when it is not the requirement is that {p[2]}"
    if style == 4:
        if op == "not":
            return f"[NOT: {p[0]}]"
        if op in ("and", "or"):
            return f"[{'ALL' if op == 'and' else 'ANY'} of: {p[0]} | {p[1]}]"
        if op == "unless":
            return f"[{p[0]} UNLESS {p[1]}]"
        return f"[IF {p[0]} THEN {p[1]} ELSE {p[2]}]"
    if op == "not":
        return f"NOT ({p[0]})" if style == 0 else f"it is not the case that ({p[0]})"
    if op in ("and", "or"):
        if style == 0:
            return f"({p[0]}) {op.upper()} ({p[1]})"
        connector = "both" if op == "and" else "at least one of"
        return f"{connector} these conditions hold: [({p[0]}); ({p[1]})]"
    if op == "unless":
        return f"({p[0]}) holds and the exception ({p[1]}) does not hold"
    return f"if ({p[0]}), use ({p[1]}); otherwise use ({p[2]})"


def skeleton(tree) -> str:
    if isinstance(tree, int):
        return "_"
    op, *children = tree
    parts = [skeleton(c) for c in children]
    if op in ("and", "or"):
        parts = sorted(parts)
    return f"{op}({','.join(parts)})"


def sort_commutative(tree):
    if isinstance(tree, int):
        return tree
    op, *children = tree
    children = [sort_commutative(c) for c in children]
    if op in ("and", "or"):
        children = sorted(children, key=skeleton)
    return (op, *children)


def relabel(tree):
    mapping: dict[int, int] = {}

    def walk(t):
        if isinstance(t, int):
            mapping.setdefault(t, len(mapping))
            return mapping[t]
        return (t[0], *[walk(c) for c in t[1:]])

    return walk(tree)


def canonical(tree) -> str:
    def show(t):
        if isinstance(t, int):
            return str(t)
        return f"{t[0]}({','.join(show(c) for c in t[1:])})"

    return show(relabel(sort_commutative(tree)))


def push_negation(tree):
    if isinstance(tree, int):
        return tree
    op, *children = tree
    if op == "not":
        inner = children[0]
        if isinstance(inner, int):
            return tree
        iop, *ic = inner
        if iop == "not":
            return push_negation(ic[0])
        if iop in ("and", "or"):
            return ("or" if iop == "and" else "and", *[push_negation(("not", c)) for c in ic])
        if iop == "unless":
            return push_negation(("or", ("not", ic[0]), ic[1]))
        return ("not", push_negation(inner))
    if op == "unless":
        return ("and", push_negation(children[0]), push_negation(("not", children[1])))
    return (op, *[push_negation(c) for c in children])


def structure_keys(tree) -> set[str]:
    return {canonical(tree), canonical(push_negation(tree))}


HELD_OUT_KEYS = set().union(*(structure_keys(SHAPES[s]) for s in DEV_SHAPES + TEST_SHAPES))


def random_tree(rng: random.Random, depth: int, next_leaf):
    if depth == 0 or (depth < 3 and rng.random() < 0.25):
        return next_leaf()
    op = rng.choice(["and", "or", "not", "unless", "if", "and", "or"])
    if op == "not":
        return ("not", random_tree(rng, depth - 1, next_leaf))
    if op == "if":
        return ("if", random_tree(rng, depth - 1, next_leaf), random_tree(rng, depth - 1, next_leaf), random_tree(rng, depth - 1, next_leaf))
    return (op, random_tree(rng, depth - 1, next_leaf), random_tree(rng, depth - 1, next_leaf))


def sample_trees(n: int, seed: Any, min_leaves: int = 2, max_leaves: int = 4, exclude: set[str] = HELD_OUT_KEYS) -> list:
    rng = random.Random(f"{seed}:trees")
    trees, keys = [], set()
    for _ in range(20000):
        if len(trees) >= n:
            break
        counter = [0]

        def next_leaf():
            counter[0] += 1
            return counter[0] - 1

        t = random_tree(rng, rng.choice([2, 2, 3]), next_leaf)
        if isinstance(t, int) or not min_leaves <= counter[0] <= max_leaves:
            continue
        k = structure_keys(t)
        if k & exclude or k & keys:
            continue
        keys |= k
        trees.append(relabel(t))
    if len(trees) < n:
        raise ValueError(f"only {len(trees)} distinct structures found")
    return trees


def leaf_indices(tree) -> set[int]:
    if isinstance(tree, int):
        return {tree}
    return set().union(*(leaf_indices(c) for c in tree[1:]))


def make_atoms(tree, rng: random.Random) -> list[dict]:
    nouns = rng.sample(NOUNS, 4)
    atoms = []
    for i in range(max(leaf_indices(tree)) + 1):
        kind = rng.choice(KINDS)
        prefix = nouns[i]
        fields = [f"{prefix} value"]
        if kind == "match":
            fields = [f"{prefix} signer", f"{prefix} designated approver"]
        elif kind == "elapsed":
            fields = [f"{prefix} start date", f"{prefix} end date"]
        elif kind == "flag":
            fields = [f"{prefix} verified"]
        atoms.append({"kind": kind, "fields": fields, "threshold": rng.randint(5, 60)})
    return atoms


def fact_domains(atoms: list[dict], rng: random.Random) -> dict[str, list]:
    domains: dict[str, list] = {}
    for a in atoms:
        kind, fs, t = a["kind"], a["fields"], a["threshold"]
        if kind == "match":
            people = rng.sample(PEOPLE, 3)
            domains.update({k: people for k in fs})
        elif kind == "elapsed":
            start = date(2027, rng.randint(1, 8), rng.randint(1, 28))
            domains[fs[0]] = [start.isoformat()]
            domains[fs[1]] = [(start + timedelta(days=n)).isoformat() for n in (max(0, t - 1), t, t + 1, t + 10)]
        elif kind == "flag":
            domains[fs[0]] = [False, True]
        else:
            domains[fs[0]] = [t - 1, t, t + 1, t + 10, t + 11]
    domains["routing reference"] = [rng.randint(100, 500), rng.randint(501, 999)]
    return domains


def rendered_facts(facts: dict, order: list[str]) -> list[str]:
    def value(v):
        return "yes" if v is True else "no" if v is False else str(v)

    return [f"The {k} is {value(facts[k])}." for k in order]


def decisive_edit(tree, atoms, domains, rng):
    for _ in range(1000):
        facts = {k: rng.choice(v) for k, v in domains.items()}
        label = evaluate_rule(tree, atoms, facts)
        keys = list(domains)
        rng.shuffle(keys)
        for key in keys:
            for value in domains[key]:
                edited = {**facts, key: value}
                if evaluate_rule(tree, atoms, edited) != label:
                    missing = {k: v for k, v in facts.items() if k != key}
                    if evaluate_rule(tree, atoms, missing) is None:
                        return facts, edited, key
    raise ValueError("cannot create a decisive edit")


def generate(groups_per_shape: int, seed: Any, shapes=TRAIN_SHAPES, styles=(0, 1), source: str = "compositional",
             trees: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records = []
    shape_trees = trees if trees is not None else {s: SHAPES[s] for s in shapes}
    for shape, tree in shape_trees.items():
        rng = random.Random(f"{seed}:{shape}")
        for i in range(groups_per_shape):
            atoms = make_atoms(tree, rng)
            domains = fact_domains(atoms, rng)
            facts, edited, deciding = decisive_edit(tree, atoms, domains, rng)
            nuisance = {**facts, "routing reference": next(v for v in domains["routing reference"] if v != facts["routing reference"])}
            order = list(facts)
            rng.shuffle(order)
            style = rng.choice(styles)
            policy = POLICY_WRAPPERS[style].format(rule=render_rule(tree, atoms, style))
            keys = ["accept", "reject"]
            rng.shuffle(keys)
            criteria = {k: "The policy permits this case" if k == "accept" else "The policy does not permit this case" for k in keys}
            group = f"composition/{seed}/{shape}/{i}"
            for kind, a, b in (("relevant", facts, edited), ("irrelevant", facts, nuisance)):
                for sibling, values in (("a", a), ("b", b)):
                    result = evaluate_rule(tree, atoms, values)
                    state = {"policy": policy, "case": " ".join(rendered_facts(values, order))}
                    records.append({
                        "state": state,
                        "questions": {"decision": {"type": "choice", "instructions": "Apply the policy to this case.",
                                                   "criteria": dict(criteria), "label": "accept" if result else "reject",
                                                   "src": f"composition_{shape}"}},
                        "_meta": {"id": f"{group}/{kind}/{sibling}", "group_id": group, "source": source, "variant": "clean",
                                  "pair_id": f"{group}/{kind}", "sibling": sibling, "pair_kind": kind, "family": shape,
                                  "family_id": shape, "render_style": style, "structure": canonical(tree),
                                  "repo": None, "revision": None, "split": "generated", "row": f"{shape}/{i}",
                                  "text_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                                  "row_sha256": hashlib.sha256(json.dumps([state, kind, sibling], sort_keys=True).encode()).hexdigest(),
                                  "certificate": {"tree": tree, "atoms": atoms, "facts": values, "order": order,
                                                  "deciding_field": deciding, "label": result}}})
    return records


def check_group(records: list[dict]) -> bool:
    if len(records) != 4:
        raise ValueError("a composition group requires four records")
    for r in records:
        c = r["_meta"]["certificate"]
        if evaluate_rule(c["tree"], c["atoms"], c["facts"]) != c["label"]:
            raise ValueError("incorrect certificate label")
        if r["state"]["case"] != " ".join(rendered_facts(c["facts"], c["order"])):
            raise ValueError("rendered facts differ from certificate")
        if r["questions"]["decision"]["label"] != ("accept" if c["label"] else "reject"):
            raise ValueError("answer differs from certificate")
    for a, b in (records[:2], records[2:]):
        ca, cb = a["_meta"]["certificate"], b["_meta"]["certificate"]
        if ca["order"] != cb["order"] or a["state"]["policy"] != b["state"]["policy"]:
            raise ValueError("pair changes more than one fact")
        if sum(ca["facts"][k] != cb["facts"][k] for k in ca["facts"]) != 1:
            raise ValueError("pair must change exactly one fact")
        expected_flip = a["_meta"]["pair_kind"] == "relevant"
        if (ca["label"] != cb["label"]) != expected_flip:
            raise ValueError("incorrect intervention label")
        missing = {k: v for k, v in ca["facts"].items() if k != ca["deciding_field"]}
        if expected_flip and evaluate_rule(ca["tree"], ca["atoms"], missing) is not None:
            raise ValueError("deciding evidence ablation did not remove the answer")
    return True
