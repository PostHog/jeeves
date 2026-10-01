# Jeeves – Reasoning improves Jev-like decision models

A reasoning Jev-style classifier with a diffusion drafter, trained with SFT and CISPO.

<img src="assets/smug.png" alt="Jeeves" width="220">

<p>
  <a href="https://huggingface.co/PostHog/jeeves"><img alt="Weights: 9B" src="https://img.shields.io/badge/WEIGHTS-9B-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

## Acknowledgements

Inspired by [Kev](https://github.com/jaredpalmer/kev).

## Highlights

- A 9B Jev-like model (Qwen3.5-9B, LoRA, pointer head) that thinks before it decides, with a block-4 diffusion drafter and the full training code and train/dev/test data.
- Beats Kev-9B and Jev on test data it was never trained on (0.889 vs 0.822 and 0.857) and on JevBench's public tiers (0.935 vs 0.866 for Jev).
- Supports yes/no (`noul`), multiple-choice (`choice`), and rating (`score`) questions in the same request, through a Jev-compatible API.
- About 0.3 s per request without thinking and a 3.3 s median with it on one H100 with `--precision fp8`. Can be sped up by truncating chain length.
- Runs on CUDA in bf16, or with FP8 linear layers (`--precision fp8`) on GPUs with compute capability 8.9 or higher. Inference also runs on Apple Silicon (MPS) in bf16 or FP8.

## Problem

Jev-like models give calibrated decision probabilities, but at low accuracy. A lot of pipelines therefore rely on a reasoning model as a fallback. Jeeves trains a Jev-like Qwen3.5-9B (LoRA and a pointer head) using CISPO to reason before it decides.

This results in better performance on out of domain tasks, and outperforms Jev in JevBench hard (public).

## Results

Accuracy with thinking, greedy, 2,560-token cap. The Kev-9B and Jev columns are the numbers Kev publishes.

| bench                                                        | Kev-9B                  | Jev                     | Jeeves                  |
| ------------------------------------------------------------ | ----------------------- | ----------------------- | ----------------------- |
| **Test overall** (out-of-domain and held-out, item-weighted) | 0.822                   | 0.857                   | <ins><b>0.889</b></ins> |
| **Transfer overall** (MMLU-Pro and buried state)             | 0.579                   | <ins><b>0.800</b></ins> | 0.746                   |
| **JevBench overall** (231 public items)                      | 0.715\*                 | 0.866                   | <ins><b>0.935</b></ins> |
| QNLI                                                         | <ins><b>0.925</b></ins> | <ins><b>0.925</b></ins> | 0.913                   |
| SciQ                                                         | 0.963                   | 0.988                   | <ins><b>0.991</b></ins> |
| TweetEval offensive                                          | 0.775                   | <ins><b>0.813</b></ins> | <ins><b>0.813</b></ins> |
| PAWS                                                         | 0.763                   | 0.788                   | <ins><b>0.875</b></ins> |
| MMLU                                                         | 0.738                   | <ins><b>0.900</b></ins> | 0.793                   |
| Emotion                                                      | 0.600                   | 0.588                   | <ins><b>0.647</b></ins> |
| Held-out rule structures                                     | 0.896                   | 0.885                   | <ins><b>1.000</b></ins> |
| Contrastive policies                                         | 0.900                   | 0.963                   | <ins><b>1.000</b></ins> |
| MMLU-Pro (10-way)                                            | 0.515                   | <ins><b>0.840</b></ins> | 0.739                   |
| Buried state                                                 | 0.740                   | 0.700                   | <ins><b>0.759</b></ins> |
| Unknowable answered at p ≥ 0.9 (lower is better)             | <ins><b>0.000</b></ins> | 0.090                   | 0.055                   |
| JevBench hard (111 public items)                             | 0.451\*                 | 0.730                   | <ins><b>0.865</b></ins> |
| JevBench ECE (public items)                                  |                         | 0.049                   | <ins><b>0.037</b></ins> |

\* No Kev-9B JevBench result is published. These are Kev-8B (Qwen3).

All JevBench numbers are on the public easy, standard and hard tiers (231 items). The sealed judge tier is not included, and the Jev and Kev numbers are restricted to the same public items.

Without thinking the same checkpoint scores 0.804 on our test split (2,962 items), against 0.840 with it.

## Quickstart

Requirements: Python 3.12 and a CUDA GPU. Inference also runs on an Apple Silicon Mac with 48 GB or more.

```bash
pip install -r requirements.txt
```

Download the released weights and serve them:

```bash
hf download PostHog/jeeves --local-dir jeeves-weights
python -m inference.serve --model jeeves-weights --drafter jeeves-weights/drafter_k4.safetensors --port 8009
```

On a Mac, the engine runs on MPS. The weights use 21 GB in bf16. The default caches (`--max-rows 8 --max-len 8192`) use another 28 GB, so a 48 GB Mac starts to swap. Use smaller caches:

```bash
python -m inference.serve --model jeeves-weights --drafter jeeves-weights/drafter_k4.safetensors --max-rows 4 --max-len 4096 --port 8009
```

On an M4 Pro, one question thinks at about 20 tokens per second, and a request without thinking takes 0.3 to 0.5 s.

`--precision fp8` quantizes the linear layers as on CUDA and runs them with a Metal w8a16 kernel. Activations stay unquantized at every size, as on CUDA. The weights then use 11.5 GB, and one question thinks at about 31 tokens per second. The outputs change slightly; on dev questions, accuracy and NLL did not change measurably.

[PostHog/jeeves-fp8](https://huggingface.co/PostHog/jeeves-fp8) holds the same FP8 weights already quantized, so the download is about half the size. They give the same outputs as `--precision fp8` on the bf16 weights:

```bash
hf download PostHog/jeeves-fp8 --local-dir jeeves-fp8
python -m inference.serve --model jeeves-fp8 --drafter jeeves-fp8/drafter_k4.safetensors --precision fp8 --max-rows 4 --max-len 4096 --port 8009
```

`python -m model.metal_test` checks the Metal kernels against float64 and eager references, and on CUDA `python -m model.triton_kernels_test` checks the Triton rotary, delta-state advance and delta gates bit for bit against the eager and fla paths, and `python -m inference.cuda_test` checks the merged projections and the attention masks. `python speed.py --model jeeves-weights --data data/dev.jsonl` times a fixed set of dev requests; `python -m prep.prep` builds `data/`.

Or fuse your own trained checkpoint into a standalone model and serve it with a drafter:

```bash
python export.py runs/cispo/final --out runs/fused
python -m inference.serve --model runs/fused --drafter runs/drafter_k4/drafter.safetensors --port 8009
```

`python export_fp8.py runs/fused --drafter runs/drafter_k4/drafter.safetensors --out runs/fused-fp8` writes the FP8 weights of an export and its drafters, in the format of PostHog/jeeves-fp8.

Then send a request in Jev's format:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  },
  "options": {"max_think": 512}}'
```

Response on one H100 with `--precision fp8`, with the three questions thinking in parallel:

```json
{
    "model": "jeeves-latest",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.19,
            "probabilities": { "returns": 0.4, "shipping": 0.14, "billing": 0.46 }
        },
        "escalate": { "type": "noul", "noul": 0.72 },
        "frustration": {
            "type": "score",
            "score": 1.5,
            "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
            "probabilities": { "0": 0.04, "1": 0.43, "2": 0.54 },
            "confidence": 0.75
        }
    },
    "usage": { "input_tokens": 129, "output_tokens": 160, "reasoning_tokens": 1536 },
    "latency_ms": 8141.6
}
```

### Python

`sdk/` is a drop-in replacement for Jev's Python SDK (`typesafe-sdk`):

```bash
pip install ./sdk
```

```python
from jeeves_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    result = client.system_one(
        state="I was charged twice. Please help.",
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
            "urgency": Score(instructions="How urgent is this?", criteria=["can wait", "this week", "today"]),
        },
        max_think=768,
        return_reasoning=True,
    )
    print(result.nouls["billing"].noul, result.choices["tone"].choice, result.scores["urgency"].score)
    print(result.reasoning["tone"].text)
```

The client connects to `http://127.0.0.1:8009` by default (or `JEEVES_BASE_URL`), needs no API key, and waits up to 120s.

### Options

`options` is optional and ignored by Jev clients that don't send it. Server-wide defaults are set with the matching `serve` flags.

| option              | default | effect                                                                       |
| ------------------- | ------- | ---------------------------------------------------------------------------- |
| `think`             | `true`  | `false` answers from the prompt alone (about 0.3 s)                          |
| `max_think`         | 2560    | truncates each reasoning chain at this many tokens, then answers             |
| `nothink_threshold` | `null`  | answers without thinking when the no-think confidence is at least this value |
| `return_reasoning`  | `false` | adds each question's reasoning text to the response                          |

On 325 dev questions, on one H100 with `--precision fp8`:

| setting                                  | accuracy | mean reasoning tokens | median / p90 latency |
| ---------------------------------------- | -------- | --------------------- | -------------------- |
| full thinking                            | 0.825    | 1,138                 | 3.3 s / 17.1 s       |
| `max_think` 768, `nothink_threshold` 0.9 | 0.806    | 344                   | 2.0 s / 5.6 s        |
| no thinking                              | 0.775    | 0                     | about 0.3 s          |

## How it works

Questions, states and answers are loaded into the Qwen chat template like

```text
<state> …state…
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> …
<think>
```

The model then rolls out its reasoning chain, and after the `</think>` token we append

```text
</think>

<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> …
<decide>
```

A pointer head scores each option with a scaled dot product between a query projection of the hidden state at `<decide>` and a key projection of the hidden state at that option's `</opt>`, where

```text
<state>, <q>, <opt>, </opt>, <decide> = "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"
```

These are rare, largely unused tokens in the Qwen tokenizer. Ablations found that using plain text like "State" in the prompt instead worsened performance.
Likewise, not repeating the questions after the reasoning block also decreases performance.
The final probabilities are a softmax over the option scores, divided by a temperature fitted on the dev set.

### Training

1. **SFT** (2 epochs, 596 steps on 8 GPUs). LoRA r=16 on all projections of Qwen3.5-9B plus the pointer head, trained on 19,126 questions from 12 public datasets and synthetic policy data. Half the questions carry a reasoning chain sampled from the base model.
2. **CISPO** (a 624-step schedule stopped at step 402). 9,992 RL questions, 8 rollouts each at temperature 1, capped at 2,560 thinking tokens.
3. **Calibration**. A single temperature fitted on dev, stored with the checkpoint.

Stopping at step 402 keeps the best calibration and dev score. Past it, the head over-sharpens on the saturated RL pool.

### Diffusion drafter

A diffusion view of the frozen model (`drafter/`), inspired by [Orthrus](https://arxiv.org/abs/2605.12825).

Unlike Orthrus, which supports attention-only models, it supports Qwen3.5's Gated DeltaNet layers by letting mask tokens cross-attend to those layers' post-convolution keys and values.

|                                             | chain tokens per second |
| ------------------------------------------- | ----------------------- |
| plain graphed greedy decoding, one question | 109                     |
| block 4, one question                       | 176 (1.6×)              |
| block 8, one question                       | 193 (1.76×)             |
| block 4, eight questions batched            | about 960 in total      |

Block 4 is the default because it stays cheap when several questions are batched.

## Reproduce

### Data

You can build the datasets locally using the prep scripts. This downloads the public datasets from Hugging Face at the revisions pinned in `prep/public.py`:

```bash
python -m prep.prep
```

Each public dataset stays under its own license.

### Training

On 8 GPUs, with the data in `data/`, `bash run.sh` runs the whole pipeline:

```bash
torchrun --nproc_per_node 8 train.py sft --run-dir runs/sft
torchrun --nproc_per_node 8 train.py cispo --run-dir runs/cispo --init runs/sft/final
torchrun --nproc_per_node 8 test.py runs/cispo/final
torchrun --nproc_per_node 8 jevbench.py runs/cispo/final
python export.py runs/cispo/final --out runs/fused
torchrun --nproc_per_node 8 -m drafter.gen --model runs/fused
torchrun --nproc_per_node 8 train.py drafter --model runs/fused --block 4 --run-dir runs/drafter_k4
```

## Repository

| path                                     | contents                                                                            |
| ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `model/`                                 | Qwen3.5 (Gated DeltaNet + gated attention), LoRA, pointer head, Metal kernels       |
| `loader/`                                | prompt format, tokenisation and batching                                            |
| `prep/`                                  | dataset construction (`prep.py`) and synthetic generators                           |
| `trainer.py`, `train.py`                 | SFT, CISPO and drafter training                                                     |
| `test.py`, `jevbench.py`, `calibrate.py` | evaluation, JevBench, temperature fitting                                           |
| `export.py`                              | fuses LoRA into a standalone model with the head and temperature                    |
| `export_fp8.py`                          | quantizes an exported model and its drafters to FP8 weights                         |
| `drafter/`                               | drafter model, chain sampling, fused speculative decoder                            |
| `inference/`                             | FP8 linears (CUDA, Metal), batched speculative engine, server and benchmark         |
| `speed.py`                               | fixed speed and equivalence harness for MPS                                         |
| `sdk/`                                   | `jeeves_sdk`, a drop-in replacement for Jev's Python SDK with the reasoning options |

## Limitations

- Knowledge questions trail Jev (MMLU 0.793 vs 0.900, MMLU-Pro 0.739 vs 0.840).
- Thinking is slow at the tail: 17 s at p90 with full chains on one H100. Use `max_think` and `nothink_threshold` when latency matters.
- The Kev and Jev comparisons outside JevBench use different items from the same sources.
- No language consistency reward was included so thinking chains are not well interpretable.

## Quote this

If you use Jeeves, its training recipe or its drafter, please cite:

```bibtex
@software{waltz2026jeeves,
  author = {Waltz, Nicholas P.},
  title  = {Jeeves: Reasoning Improves Jev-like Decisions},
  year   = {2026},
  url    = {https://github.com/PostHog/jeeves},
  note   = {Qwen3.5-9B decision model trained with SFT and CISPO, with a block-4 diffusion drafter}
}
```

## References

- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked), the Jev design that Kev and Jeeves follow.
- Qwen Team. [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), the base model.
- Yang, Kautz, Hatamizadeh. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). ICLR 2025.
- Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- MiniMax. [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585). 2025. Introduces CISPO.
- Guo, Pleiss, Sun, Weinberger. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). ICML 2017. Temperature scaling.
- [Orthrus](https://arxiv.org/abs/2605.12825), arXiv 2605.12825. The diffusion drafter ours adapts to Gated DeltaNet.
- Leviathan, Kalman, Matias. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). ICML 2023.
