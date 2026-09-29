#!/usr/bin/env bash
set -euo pipefail

NPROC="${NPROC:-8}"
TR="torchrun --nproc_per_node $NPROC"

$TR train.py sft --run-dir runs/sft
$TR train.py cispo --run-dir runs/cispo --init runs/sft/final
$TR test.py runs/cispo/final
$TR jevbench.py runs/cispo/final
python export.py runs/cispo/final --out runs/fused
$TR -m orthrus.gen --model runs/fused
$TR train.py orthrus --model runs/fused --block 4 --run-dir runs/orthrus_k4
