"""Tests for prep.contrast_cases: the variants change a question's option set, so a
soft target inherited from the source record has to be rebuilt to match."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prep.format import Question
from prep.prep import NONE_KEY, contrast_cases

TARGET = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def record(label="b", target=None):
    q = {"type": "choice", "instructions": "Which is needed?", "criteria": {"a": "alpha", "b": "beta", "c": "gamma"},
         "label": label, "src": "unknowable_control"}
    if target is not None:
        q["target"] = target
    return {"state": "s", "questions": {"decision": q}, "_meta": {"id": "r1", "source": "unknowable_control",
                                                                "group_id": "g1"}}


def variant(variants, name):
    (rec,) = [v for v in variants if v["_meta"]["variant"] == name]
    return Question.from_dict("decision", rec["questions"]["decision"])


def test_none_absent_target_puts_the_removed_options_mass_on_the_declared_answer():
    # Regression: the variant popped "b" and declared none_of_these correct, but inherited a
    # target still keyed {a, b, c}. target_vector() scored none_of_these at 0.0 and spread the
    # rest over the distractors, so the item trained against itself.
    variants = contrast_cases(record(target=TARGET), 0)
    q = variant(variants, "none_absent")
    assert q.label == NONE_KEY
    vector = q.target_vector()
    assert vector[q.keys().index(NONE_KEY)] > 0.0
    assert sum(vector) == pytest.approx(1.0)
    # "b" is gone, so it must carry nothing, and the surviving options keep the mass it had:
    # a, c and none_of_these each end at 1/3. (Unfixed, none_of_these scored 0.0 and a and c
    # were renormalised to 0.5 each.)
    assert "b" not in q.keys()
    assert vector[q.keys().index("a")] == pytest.approx(1 / 3)
    assert vector[q.keys().index("c")] == pytest.approx(1 / 3)
    assert vector[q.keys().index(NONE_KEY)] == pytest.approx(1 / 3)


def test_none_present_target_gives_the_new_key_no_mass():
    # Nothing was removed and none_of_these is only a distractor here, so it takes no mass and
    # the original answer keeps all of its own.
    variants = contrast_cases(record(target=TARGET), 0)
    q = variant(variants, "none_present")
    assert q.label == "b"
    vector = q.target_vector()
    assert vector[q.keys().index(NONE_KEY)] == 0.0
    assert vector[q.keys().index("b")] == pytest.approx(1 / 3)
    assert sum(vector) == pytest.approx(1.0)


def test_a_labelled_target_is_transferred_whole_to_the_replacement_key():
    sharp = {"a": 0.0, "b": 1.0, "c": 0.0}
    variants = contrast_cases(record(target=sharp), 0)
    q = variant(variants, "none_absent")
    vector = q.target_vector()
    assert vector[q.keys().index(NONE_KEY)] == pytest.approx(1.0)
    assert vector[q.keys().index("a")] == 0.0


def test_records_without_a_target_are_untouched():
    variants = contrast_cases(record(), 0)
    for name in ("none_present", "none_absent"):
        assert variant(variants, name).target_vector() is None


def test_the_permuted_variant_keeps_the_original_option_set_and_target():
    variants = contrast_cases(record(target=TARGET), 0)
    (rec,) = [v for v in variants if v["_meta"]["variant"] == "permuted"]
    q = Question.from_dict("decision", rec["questions"]["decision"])
    assert set(q.criteria) == {"a", "b", "c"}
    assert q.target_vector() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
