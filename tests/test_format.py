"""Tests for the record format: question validation, rendering and JSON round-trips."""

from __future__ import annotations

import json

import pytest

from prep.format import CHOICE, NOUL, SCORE, DataFormat, Question, option_text, read_jsonl, render, write_jsonl


def test_noul_criteria_rejects_keys_other_than_true_and_false() -> None:
    # options() reads criteria["false"]/criteria["true"], so any other key is
    # dropped on the floor and the question silently degrades to bare "no"/"yes".
    with pytest.raises(ValueError, match="may only describe true and false"):
        Question(id="escalate", type=NOUL, instructions="Urgent?", criteria={"yes": "urgent", "no": "routine"})


def test_noul_criteria_names_the_offending_keys() -> None:
    with pytest.raises(ValueError) as excinfo:
        Question(id="escalate", type=NOUL, instructions="Urgent?", criteria={"yes": "urgent", "True": "urgent"})
    assert "'True'" in str(excinfo.value)
    assert "'yes'" in str(excinfo.value)


def test_noul_criteria_still_requires_a_dict_or_null() -> None:
    with pytest.raises(ValueError, match="must be a dict or null"):
        Question(id="escalate", type=NOUL, instructions="Urgent?", criteria=["yes", "no"])


def test_noul_criteria_accepts_true_false_and_null() -> None:
    for criteria in (None, {}, {"true": "needs a human"}, {"false": "on its own", "true": "needs a human"}):
        q = Question(id="escalate", type=NOUL, instructions="Urgent?", criteria=criteria)
        assert q.keys() == ["false", "true"]


def test_noul_options_keep_their_descriptions_in_false_true_order() -> None:
    q = Question(id="escalate", type=NOUL, instructions="Urgent?",
                 criteria={"true": "needs a human", "false": "can wait"})
    assert q.options() == ["no: can wait", "yes: needs a human"]


def test_from_dict_rejects_a_noul_with_unknown_criteria_keys(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({
        "state": "hello",
        "questions": {"escalate": {"type": "noul", "instructions": "Urgent?",
                                   "criteria": {"yes": "urgent", "no": "routine"}}},
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_jsonl(path)
    assert "may only describe true and false" in str(excinfo.value)
    assert f"{path}:1" in str(excinfo.value)


def test_round_trip_keeps_a_valid_noul(tmp_path) -> None:
    record = DataFormat.from_dict({
        "state": {"ticket": "refund requested"},
        "questions": {
            "escalate": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "needs a human"}},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "returns": "refunds"}},
            "tone": {"type": "score", "instructions": "How annoyed?", "criteria": ["calm", "annoyed"]},
        },
    })

    path = tmp_path / "data.jsonl"
    assert write_jsonl(path, [record]) == 1
    (loaded,) = read_jsonl(path)

    assert loaded.state == record.state
    assert [q.type for q in loaded.questions] == [NOUL, CHOICE, SCORE]
    assert loaded.question("escalate").criteria == {"true": "needs a human"}
    assert loaded.question("team").keys() == ["billing", "returns"]
    assert loaded.question("tone").keys() == ["0", "1"]


def test_option_text_passes_through_empty_descriptions() -> None:
    assert option_text("billing", None) == "billing"
    assert option_text("billing", "") == "billing"
    assert option_text("billing", "charges") == "billing: charges"


def test_render_leaves_scalars_alone() -> None:
    assert render(None) == ""
    assert render("plain") == "plain"
    assert render(7) == "7"
    assert render(True) == "True"
