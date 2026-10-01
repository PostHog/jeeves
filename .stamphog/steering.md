# Jeeves

Jeeves is a research repository for a 9B decision model.
It contains training code, data preparation, an inference server, and a Python SDK.
Most changes are contained and easy to revert, and the hosted guidance applies to them as written.

Three areas are risky territory in this repository, although the hosted guidance does not name them.
Apply the risky-territory and independent-assurance rules to them.

## Model outputs

A change in these paths can change the answers of a released model without an error:

- `model/`
- `inference/engine.py` and `inference/fp8*.py`
- `export.py`
- `prep/format.py`, which renders the prompt for training and for inference

A speedup must keep the outputs the same.
The PR must show evidence of this: `python -m model.metal_test` for a Metal kernel, the `speed.py` equivalence check for MPS, or a before-and-after comparison on dev data.
If a speedup has no evidence, ESCALATE.

The default precision is bf16.
FP8 and every other lossy mode must stay opt-in.
A change that makes a lossy mode the default, or that changes the outputs in the default mode, needs independent assurance.

## Training and evaluation data

`data/manifest.json` sets the seed, the tokenizer, the dataset revisions, and which sources are trainable and which are eval-only.
The `.json` allow-list puts this file in the T0 tier, but do not use the lighter T0 bar for it.
If a source moves from `eval_only_sources` to `trainable_sources`, the test data leaks into training.
REFUSE that change unless the description explains why.

`prep/`, `test.py`, `jevbench.py` and `metrics.py` produce the numbers that the README reports.
Treat a change to how they split, filter, or score data as risky territory.

The results and speed tables in `README.md` are measurements.
A PR that changes a number in them must say which run produced the new number.
If it does not, ESCALATE.

## Jev-compatible API and SDK

`inference/api.py`, `inference/types.py` and `sdk/` implement a Jev-compatible API.
`jeeves_sdk` is a drop-in replacement for the `typesafe-sdk` package of Jev.
Treat the request and response shapes as a public API contract.

A new optional field is additive and is not risky.
A change that removes or renames a field, changes a type or a default, or changes an error, breaks clients.
ESCALATE such a change if it has no independent assurance.
