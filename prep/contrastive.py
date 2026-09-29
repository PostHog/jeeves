from __future__ import annotations

import hashlib
import json
import random
from datetime import date, timedelta
from typing import Any, Callable

UNDETERMINED = "UNDETERMINED"
NAMES = ["Mira", "Noah", "Priya", "Tomas", "Aiko", "Lena", "Omar", "Sana", "Jonas", "Ravi", "Elin", "Kofi"]
ROLES = ["account owner", "billing manager", "support agent", "warehouse lead"]
ITEMS = ["a pair of running shoes", "a desk lamp", "a wireless keyboard", "a rain jacket", "a coffee grinder", "a backpack"]
PROGRAMS = ["the volunteer driver program", "the apprenticeship", "the rental agreement", "the night-shift roster"]


def day(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def need(facts: dict, *keys: str) -> bool:
    return all(k in facts for k in keys)


def item(policy: str, sentences: list, evaluate: Callable, question: dict) -> dict[str, Any]:
    return {"policy": policy, "sentences": sentences, "evaluate": evaluate, "question": question}


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def family_return_window(rng):
    window = rng.choice([14, 30, 45, 60])
    thing, name = rng.choice(ITEMS), rng.choice(NAMES)
    bought = date(2026, rng.randint(1, 9), rng.randint(1, 28))

    def evaluate(f):
        return (f["request"] - f["purchase"]).days <= window if need(f, "request", "purchase") else UNDETERMINED

    def build(days):
        request = bought + timedelta(days=days)
        return item(f"Returns are accepted only if the return request is submitted within {window} days of the purchase date.",
                    [(f"{name} bought {thing} on {day(bought)}.", {"purchase": bought}),
                     (f"The return request was submitted on {day(request)}.", {"request": request}),
                     (f"The order was paid by card and shipped to {name}'s home address.", {})],
                    evaluate, noul("Is this return request within the policy window?"))

    return build(rng.randint(1, window - 1)), build(window + rng.randint(1, 30))


def family_spend_threshold(rng):
    limit, name, role = rng.choice([250, 500, 1000, 2500]), rng.choice(NAMES), rng.choice(ROLES)

    def evaluate(f):
        return ("auto_approved" if f["amount"] <= limit else "director_signoff") if need(f, "amount") else UNDETERMINED

    def build(amount):
        return item(f"Expense claims of ${limit:,} or less are approved automatically. Claims above ${limit:,} require director sign-off.",
                    [(f"{name}, the {role}, submitted an expense claim.", {}),
                     (f"The claim total is ${amount:,}.", {"amount": amount}),
                     ("Receipts were attached for every line item.", {})],
                    evaluate, choice("How is this claim handled under the policy?",
                                     {"auto_approved": "Approved without further review", "director_signoff": "Requires director sign-off",
                                      "rejected": "Rejected outright"}))

    return build(limit - rng.randint(1, limit // 2)), build(limit + rng.randint(1, limit))


def family_authorization(rng):
    approver, other = rng.sample(NAMES, 2)
    account, amount = rng.randint(10, 99), rng.choice([40, 120, 350, 900])

    def evaluate(f):
        return f["signer"] == f["approver"] if need(f, "signer", "approver") else UNDETERMINED

    def build(signer):
        return item("A refund is authorized only when its sole authorization was signed by someone who may authorize refunds for that account.",
                    [(f"Only {approver} may authorize refunds for account {account}.", {"approver": approver}),
                     (f"The sole authorization for this refund on account {account} was signed by {signer}.", {"signer": signer}),
                     (f"The refund amount is ${amount}.", {})],
                    evaluate, noul("Is the refund authorized?"))

    return build(approver), build(other)


def family_age_eligibility(rng):
    minimum, name, program = rng.choice([16, 18, 21, 25]), rng.choice(NAMES), rng.choice(PROGRAMS)

    def evaluate(f):
        return f["age"] >= minimum if need(f, "age") else UNDETERMINED

    def build(age):
        return item(f"Applicants must be at least {minimum} years old to be eligible for {program}.",
                    [(f"{name} applied to join {program}.", {}),
                     (f"{name} is {age} years old.", {"age": age}),
                     ("The application form was complete and signed.", {})],
                    evaluate, noul("Is the applicant eligible?"))

    return build(minimum + rng.randint(0, 20)), build(minimum - rng.randint(1, 5))


def family_quantity_limit(rng):
    limit, thing, name = rng.choice([2, 3, 5, 10]), rng.choice(ITEMS), rng.choice(NAMES)

    def evaluate(f):
        if not need(f, "qty"):
            return UNDETERMINED
        return "within_limit" if f["qty"] <= limit else "slightly_over" if f["qty"] <= 2 * limit else "far_over"

    def build(qty):
        return item(f"Customers may order at most {limit} units of any single item per order. Orders up to double the limit are held for review; larger orders are cancelled.",
                    [(f"{name} placed an order for {thing}.", {}),
                     (f"The order quantity is {qty}.", {"qty": qty}),
                     ("Delivery was requested to a residential address.", {})],
                    evaluate, choice("What happens to this order?",
                                     {"within_limit": "Processed normally", "slightly_over": "Held for review", "far_over": "Cancelled"}))

    return build(rng.randint(1, limit)), build(rng.choice([rng.randint(limit + 1, 2 * limit), rng.randint(2 * limit + 1, 4 * limit)]))


def family_deadline(rng):
    name = rng.choice(NAMES)
    due, grace = date(2026, rng.randint(2, 11), rng.randint(1, 28)), rng.choice([3, 7, 14])

    def evaluate(f):
        if not need(f, "received", "due"):
            return UNDETERMINED
        late = (f["received"] - f["due"]).days
        return 0 if late <= 0 else 1 if late <= grace else 2

    def build(offset):
        received = due + timedelta(days=offset)
        return item(f"Reports received by the deadline are on time. Reports received within {grace} days after the deadline are late but accepted. Later reports are refused.",
                    [(f"The filing deadline for {name}'s report was {day(due)}.", {"due": due}),
                     (f"The report was received on {day(received)}.", {"received": received}),
                     ("The report was submitted through the online portal.", {})],
                    evaluate, score("How late is this report?", ["On time", "Late but accepted", "Refused"]))

    return build(-rng.randint(0, 10)), build(rng.choice([rng.randint(1, grace), grace + rng.randint(1, 20)]))


def family_warranty_claim(rng):
    name, thing = rng.choice(NAMES), rng.choice(ITEMS)
    bought = date(2026, rng.randint(1, 6), rng.randint(1, 28))
    standard, extended = rng.choice([(90, 365), (180, 730), (365, 1095)])
    part = rng.choice(["stitching", "battery", "housing", "zipper", "switch"])

    def evaluate(f):
        if not need(f, "claim", "purchase"):
            return UNDETERMINED
        age = (f["claim"] - f["purchase"]).days
        return 0 if age <= standard else 1 if age <= extended else 2

    def build(age):
        claim = bought + timedelta(days=age)
        return item(f"Warranty claims made within {standard} days of purchase are covered in full. Claims made after that but within {extended} days are covered at half cost. Later claims are not covered.",
                    [(f"{name} purchased {thing} on {day(bought)}.", {"purchase": bought}),
                     (f"A warranty claim for it was filed on {day(claim)}.", {"claim": claim}),
                     (f"The claim describes a defect in the {part}.", {})],
                    evaluate, score("How is this claim covered?", ["Covered in full", "Covered at half cost", "Not covered"]))

    def tier(age):
        return evaluate({"claim": bought + timedelta(days=age), "purchase": bought})

    a = rng.choice([rng.randint(1, standard), rng.randint(standard + 1, extended), extended + rng.randint(1, 200)])
    candidates = [x for x in [rng.randint(1, standard), rng.randint(standard + 1, extended), extended + rng.randint(1, 200)] if tier(x) != tier(a)]
    b = rng.choice(candidates or [a])
    return build(a), build(b)


def family_sla_response(rng):
    name = rng.choice(NAMES)
    target, breach = rng.choice([(4, 24), (8, 48), (24, 72), (1, 8)])
    topic = rng.choice(["a login failure", "a duplicate charge", "a missing invoice", "an export error"])
    queue = rng.choice(["email", "chat", "phone"])

    def evaluate(f):
        if not need(f, "hours"):
            return UNDETERMINED
        return 0 if f["hours"] <= target else 1 if f["hours"] <= breach else 2

    def build(h):
        return item(f"Support responses within {target} hours meet the service level. Responses after {target} but within {breach} hours are a minor breach. Anything slower is a major breach.",
                    [(f"{name} opened a priority ticket about {topic}.", {}),
                     (f"The first response arrived {h} hours after the ticket was opened.", {"hours": h}),
                     (f"The ticket was routed through the {queue} queue.", {})],
                    evaluate, score("How does this response time rate against the service level?", ["Met", "Minor breach", "Major breach"]))

    a, b = rng.sample([rng.randint(1, target), rng.randint(target + 1, breach), breach + rng.randint(1, 100)], 2)
    return build(a), build(b)


def family_late_fee(rng):
    name = rng.choice(NAMES)
    grace, cap = rng.choice([(5, 30), (10, 60), (15, 45)])
    amount, method = rng.choice([120, 450, 980, 2300]), rng.choice(["bank transfer", "card", "cheque"])

    def evaluate(f):
        if not need(f, "days_late"):
            return UNDETERMINED
        return 0 if f["days_late"] <= grace else 1 if f["days_late"] <= cap else 2

    def build(d):
        return item(f"Invoices paid within {grace} days after the due date incur no fee. Payments between {grace + 1} and {cap} days late incur a 2% fee. Payments later than {cap} days incur a 10% fee and a hold on the account.",
                    [(f"{name}'s invoice for ${amount:,} fell due last quarter.", {}),
                     (f"Payment was received {d} days after the due date.", {"days_late": d}),
                     (f"The payment was made by {method}.", {})],
                    evaluate, score("Which fee tier applies?", ["No fee", "2% fee", "10% fee and account hold"]))

    a, b = rng.sample([rng.randint(0, grace), rng.randint(grace + 1, cap), cap + rng.randint(1, 60)], 2)
    return build(a), build(b)


def family_volume_discount(rng):
    name, thing = rng.choice(NAMES), rng.choice(ITEMS)
    t1, t2 = rng.choice([(10, 50), (25, 100), (5, 20), (100, 500)])
    dest = rng.choice(["warehouse", "storefront", "branch office"])

    def evaluate(f):
        if not need(f, "units"):
            return UNDETERMINED
        return 0 if f["units"] < t1 else 1 if f["units"] < t2 else 2

    def build(u):
        return item(f"Orders of fewer than {t1} units are charged the list price. Orders of {t1} to {t2 - 1} units receive the volume discount. Orders of {t2} units or more receive the wholesale rate.",
                    [(f"{name} placed a business order for {thing}.", {}),
                     (f"The order is for {u} units.", {"units": u}),
                     (f"Delivery is to a {dest}.", {})],
                    evaluate, score("Which pricing tier applies?", ["List price", "Volume discount", "Wholesale rate"]))

    a, b = rng.sample([rng.randint(1, t1 - 1), rng.randint(t1, t2 - 1), t2 + rng.randint(0, t2)], 2)
    return build(a), build(b)


def family_shipping_delay(rng):
    name, thing = rng.choice(NAMES), rng.choice(ITEMS)
    promised = date(2026, rng.randint(1, 11), rng.randint(1, 28))
    minor, major = rng.choice([(1, 4), (2, 7), (3, 10)])
    carrier = rng.choice(["the courier", "the postal service", "a freight partner"])

    def evaluate(f):
        if not need(f, "delivered", "promised"):
            return UNDETERMINED
        late = (f["delivered"] - f["promised"]).days
        return 0 if late <= minor else 1 if late <= major else 2

    def build(offset):
        delivered = promised + timedelta(days=offset)
        plural = "s" if minor > 1 else ""
        return item(f"Deliveries up to {minor} day{plural} after the promised date count as on time. Deliveries {minor + 1} to {major} days after it are a minor delay and earn a shipping refund. Later deliveries are a major delay and earn a full refund.",
                    [(f"{name} ordered {thing} with delivery promised for {day(promised)}.", {"promised": promised}),
                     (f"The parcel was delivered on {day(delivered)}.", {"delivered": delivered}),
                     (f"It was shipped by {carrier}.", {})],
                    evaluate, score("How is this delivery classified?", ["On time", "Minor delay: shipping refund", "Major delay: full refund"]))

    a, b = rng.sample([rng.randint(-3, minor), rng.randint(minor + 1, major), major + rng.randint(1, 30)], 2)
    return build(a), build(b)


FAMILIES = {
    "return_window": family_return_window, "spend_threshold": family_spend_threshold, "authorization": family_authorization,
    "age_eligibility": family_age_eligibility, "quantity_limit": family_quantity_limit, "deadline": family_deadline,
    "warranty_claim": family_warranty_claim, "sla_response": family_sla_response, "late_fee": family_late_fee,
    "volume_discount": family_volume_discount, "shipping_delay": family_shipping_delay,
}
LEGACY_FAMILIES = ("return_window", "spend_threshold", "age_eligibility", "quantity_limit")
ORDINAL_FAMILIES = ("warranty_claim", "sla_response", "late_fee", "volume_discount")
TRAINABLE_FAMILIES = LEGACY_FAMILIES + ORDINAL_FAMILIES
HELD_OUT_FAMILIES = ("authorization", "deadline")
DATE_FAMILIES = ("return_window", "warranty_claim", "shipping_delay")


def label_of(it: dict, drop: int | None = None) -> Any:
    facts: dict = {}
    for i, (_, f) in enumerate(it["sentences"]):
        if i != drop:
            facts.update(f)
    return it["evaluate"](facts)


def check_pair(a: dict, b: dict) -> str | None:
    la, lb = label_of(a), label_of(b)
    if UNDETERMINED in (la, lb):
        return "label_undetermined"
    if la == lb:
        return "labels_equal"
    if len(a["sentences"]) != len(b["sentences"]) or sum(x[0] != y[0] for x, y in zip(a["sentences"], b["sentences"])) != 1:
        return "not_exactly_one_sentence_differs"
    for it, label in ((a, la), (b, lb)):
        evidence = 0
        for i, (_, facts) in enumerate(it["sentences"]):
            got = label_of(it, drop=i)
            if facts and got != UNDETERMINED:
                return "ablation_failed"
            if not facts and got != label:
                return "invariance_failed"
            evidence += bool(facts)
        if evidence < 1:
            return "no_evidence_sentence"
    return None


def to_request(it: dict, family: str, pair_id: str, sibling: str, rng: random.Random) -> dict[str, Any]:
    order = list(range(len(it["sentences"])))
    rng.shuffle(order)
    sentences = [it["sentences"][i][0] for i in order]
    q = {**it["question"], "label": label_of(it), "src": f"contrastive_{family}"}
    return {"state": {"policy": it["policy"], "case": " ".join(sentences)}, "questions": {"decision": q},
            "_meta": {"source": "contrastive", "family": family, "family_id": f"{family}/{pair_id}", "pair_id": pair_id,
                      "sibling": sibling, "repo": None, "revision": None, "split": "generated", "row": pair_id,
                      "id": f"contrastive/{family}/{pair_id}/{sibling}", "group_id": f"{family}/{pair_id}", "variant": "clean",
                      "text_sha256": hashlib.sha256(" ".join(sentences).casefold().encode()).hexdigest(),
                      "row_sha256": hashlib.sha256(json.dumps([t for t, _ in it["sentences"]]).encode()).hexdigest()}}


def valid_pairs(family: str, n_pairs: int, seed: Any):
    rng = random.Random(f"{seed}:{family}")
    kept, attempts, reasons = 0, 0, {}
    while kept < n_pairs and attempts < 50 * n_pairs:
        attempts += 1
        a, b = FAMILIES[family](rng)
        why = check_pair(a, b)
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        yield f"{seed}-{family}-{kept:04d}", a, b, rng.getrandbits(64)
        kept += 1
    if kept < n_pairs:
        raise ValueError(f"{family}: only {kept}/{n_pairs} pairs passed checks ({reasons})")


def generate(n_pairs_per_family: int, seed: Any, families: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    records = []
    for family in families or FAMILIES:
        for pair_id, a, b, order_seed in valid_pairs(family, n_pairs_per_family, seed):
            records.append(to_request(a, family, pair_id, "a", random.Random(order_seed)))
            records.append(to_request(b, family, pair_id, "b", random.Random(order_seed)))
    return records
