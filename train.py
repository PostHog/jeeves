from __future__ import annotations

import argparse
from dataclasses import fields

from orthrus.train import OrthrusConfig, train_orthrus
from trainer import Config, run

STAGES = {"sft": Config, "cispo": Config, "orthrus": OrthrusConfig}


def add_fields(ap: argparse.ArgumentParser, cls) -> None:
    defaults = cls()
    for f in fields(cls):
        if f.name == "stage":
            continue
        default = getattr(defaults, f.name)
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(default, bool):
            ap.add_argument(flag, dest=f.name, default=default, action=argparse.BooleanOptionalAction)
        else:
            ap.add_argument(flag, dest=f.name, default=default, type=type(default) if default is not None else str)


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT and CISPO for the Qwen3.5 pointer-decision model, then Orthrus drafter distillation.")
    sub = ap.add_subparsers(dest="stage", required=True)
    for stage, cls in STAGES.items():
        add_fields(sub.add_parser(stage), cls)
    args = vars(ap.parse_args())
    stage = args.pop("stage")
    if stage == "orthrus":
        train_orthrus(OrthrusConfig(**args))
    else:
        run(Config(stage=stage, **args))


if __name__ == "__main__":
    main()
