# Jeeves – Reasoning improves Jev-like decision models

A reasoning Jev-style classifier with a diffusion drafter, trained with SFT and CISPO.

<img src="assets/smug.png" alt="Jeeves" width="220">

<p>
  <a href="https://huggingface.co/collections/"><img alt="Weights: 9B" src="https://img.shields.io/badge/WEIGHTS-9B-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

## Acknowledgements

Inspired by [Kev](https://github.com/jaredpalmer/kev).

## Highlights

- A 9B Jev-like model (Qwen3.5-9B, LoRA, pointer head) that thinks before it decides, with a block-4 diffusion drafter and the full training code and train/dev/test data.
- Beats Kev-9B and Jev on out-of-domain accuracy (0.889 vs 0.822 and 0.857) and on JevBench's public tiers (0.935 vs 0.866 for Jev). Knowledge-heavy benchmarks are the exception: MMLU-Pro is 0.739, against Kev-9B's 0.515 and Jev's 0.840.
- Calibrated out of the box: a fitted temperature brings top-label calibration error to 0.021 and confident errors (wrong at p ≥ 0.9) to 1.3% of answers.
- Supports yes/no (`noul`), multiple-choice (`choice`), and rating (`score`) questions in the same request, through a Jev-compatible API.
- About 0.3 s per request without thinking and a 3.3 s median with it on one H100. Truncating chains and skipping thinking for confident questions cuts the median to 2.0 s for about one point of accuracy.
- Runs on CUDA (Hopper for the FP8 kernel).

## Problem

Jev-like models give calibrated decision probabilities, but at low accuracy. A lot of pipelines therefore rely on a reasoning model as a fallback. Jeeves trains a Jev-like Qwen3.5-9B (LoRA and a pointer head) using CISPO to reason before it decides.

This results in better performance on out of domain tasks, and outperforms Jev in JevBench hard (public).

## Results

Accuracy with thinking, greedy, 2,560-token cap. The Kev-9B and Jev columns are the numbers Kev publishes; only JevBench uses the same items for every model, so gaps under about 5 points on the other rows are within noise.

| bench                                            | Kev-9B    | Jev       | Jeeves    |
| ------------------------------------------------ | --------- | --------- | --------- |
| **Out-of-domain overall** (item-weighted)        | 0.822     | 0.857     | **0.889** |
| **Transfer overall** (MMLU-Pro and buried state) | 0.579     | **0.800** | 0.746     |
| **JevBench overall** (231 public items)          | 0.715\*   | 0.866     | **0.935** |
| QNLI                                             | **0.925** | **0.925** | 0.913     |
| SciQ                                             | 0.963     | 0.988     | **0.991** |
| TweetEval offensive                              | 0.775     | **0.813** | **0.813** |
| PAWS                                             | 0.763     | 0.788     | **0.875** |
| MMLU                                             | 0.738     | **0.900** | 0.793     |
| Emotion                                          | 0.600     | 0.588     | **0.647** |
| Held-out rule structures                         | 0.896     | 0.885     | **1.000** |
| Contrastive policies                             | 0.900     | 0.963     | **1.000** |
| MMLU-Pro (10-way)                                | 0.515     | **0.840** | 0.739     |
| Buried state                                     | 0.740     | 0.700     | **0.759** |
| Unknowable answered at p ≥ 0.9 (lower is better) | **0.000** | 0.090     | 0.055     |
| JevBench hard (111 public items)                 | 0.451\*   | 0.730     | **0.865** |
| JevBench ECE (public items)                      |           | 0.049     | **0.037** |

\* No Kev-9B JevBench result is published; these are Kev-8B (Qwen3).

All JevBench numbers are on the public easy, standard and hard tiers (231 items). The sealed judge tier is not included, and the Jev and Kev numbers are restricted to the same public items.

Without thinking the same checkpoint scores 0.804 on our test split (2,962 items), against 0.840 with it. With the fitted temperature (T = 1.859) the no-think path has a top-label ECE of 0.021 and 1.3% confident errors, against Kev-9B's 0.042 and 4.0% and Jev's 0.049 and 3.7%.

## Quickstart

Requirements: Python 3.12 and a CUDA GPU. Install the pinned versions we tested with:

```bash
pip install -r requirements.txt
```

Fuse a trained checkpoint into a standalone model and serve it with a drafter:

```bash
python export.py runs/cispo/final --out runs/fused
python -m inference.serve --model runs/fused --drafter runs/orthrus_k4/orthrus.safetensors --port 8009
```

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

Response on one H100 (FP8), with the three questions thinking in parallel:

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

`sdk/` is a drop-in replacement for Jev's Python SDK (`typesafe-sdk`). It re-exports the official classes and errors unchanged, so existing code only changes its import:

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

The client connects to `http://127.0.0.1:8009` by default (or `JEEVES_BASE_URL`), needs no API key, and waits up to 120 s. The four options below are optional keyword arguments of `system_one`; leave them out and the request is exactly what `typesafe-sdk` sends. The official `typesafe-sdk` client also works against the server with `base_url` set.

### Options

`options` is optional and ignored by Jev clients that don't send it. Server-wide defaults are set with the matching `serve` flags.

| option              | default | effect                                                                       |
| ------------------- | ------- | ---------------------------------------------------------------------------- |
| `think`             | `true`  | `false` answers from the prompt alone (about 0.3 s)                          |
| `max_think`         | 2560    | truncates each reasoning chain at this many tokens, then answers             |
| `nothink_threshold` | `null`  | answers without thinking when the no-think confidence is at least this value |
| `return_reasoning`  | `false` | adds each question's reasoning text to the response                          |

On 325 dev questions:

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
The final probabilities are a softmax over the option scores, divided by a temperature fitted on the dev set.

### Training

1. **SFT** (2 epochs, 596 steps on 8 GPUs). LoRA r=16 on all projections of Qwen3.5-9B plus the pointer head, trained on 19,126 questions from 12 public datasets and synthetic policy data. Half the questions carry a reasoning chain sampled from the base model; the loss is the pointer head's NLL at `<decide>` only.
2. **CISPO** (a 624-step schedule stopped at step 402). 9,992 RL questions, 8 rollouts each at temperature 1, capped at 2,560 thinking tokens. The reward is the probability of the right answer, discounted by up to 10% for chains longer than a hinge that moves from 2,048 to 1,024 tokens. The policy gradient runs on the chain tokens, alongside pointer-head NLL on the rollouts and on anchor questions from the SFT set. Sampling masks every special token except `</think>`.
3. **Calibration**. A single temperature fitted on dev, stored with the checkpoint.

Stopping at step 402 keeps the best calibration and JevBench score; past it, the head over-sharpens on the saturated RL pool.

### Drafter

The Orthrus drafter is a diffusion view of the same frozen model: extra query/key/value projections that let masked positions read the verified context and predict the next 3 tokens in one pass. It is distilled by KL divergence from the policy on 349k chains sampled from the trained model. Each decode cycle verifies the drafted tokens and drafts the next block in the same forward pass, so output matches greedy decoding up to numerical noise.

|                                             | chain tokens per second |
| ------------------------------------------- | ----------------------- |
| plain graphed greedy decoding, one question | 109                     |
| block 4, one question                       | 176 (1.6×)              |
| block 8, one question                       | 193 (1.76×)             |
| block 4, eight questions batched            | about 960 in total      |

Block 4 is the default because it stays cheap when several questions are batched.

### Inference engine

`inference/` prefills the shared state once for all questions in a request, then decodes every question's chain together with CUDA-graphed speculative cycles. Weights are FP8 (e4m3, per-channel scales) through a Triton kernel; FP8 matches bf16 accuracy (0.825 vs 0.818 on dev) and speed for one or two questions, and is slower than bf16 for four to eight. Pass `--no-fp8` to serve in bf16.

## Reproduce

On 8 GPUs, with the data in `data/`, `bash run.sh` runs the whole pipeline (set `NPROC` for a different GPU count). Its steps are:

```bash
torchrun --nproc_per_node 8 train.py sft --run-dir runs/sft
torchrun --nproc_per_node 8 train.py cispo --run-dir runs/cispo --init runs/sft/final
torchrun --nproc_per_node 8 test.py runs/cispo/final
torchrun --nproc_per_node 8 jevbench.py runs/cispo/final
python export.py runs/cispo/final --out runs/fused
torchrun --nproc_per_node 8 -m orthrus.gen --model runs/fused
torchrun --nproc_per_node 8 train.py orthrus --model runs/fused --block 4 --run-dir runs/orthrus_k4
```

A from-scratch run on 8×H100 took 33 minutes for SFT and 8.2 hours for CISPO, and matched the published checkpoint within noise (test thinking 0.851 vs 0.840, JevBench hard 0.838 vs 0.865).

## Repository

| path                                                 | contents                                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `model/`                                             | Qwen3.5 (Gated DeltaNet + gated attention), LoRA, pointer head                      |
| `loader/`                                            | prompt format, tokenisation and batching                                            |
| `prep/`                                              | dataset construction (`prep.py`) and synthetic generators                           |
| `trainer.py`, `train.py`                             | SFT, CISPO and drafter training                                                     |
| `test.py`, `jevbench.py`, `calibrate.py`                 | evaluation, JevBench, temperature fitting                                           |
| `export.py`                                          | fuses LoRA into a standalone model with the head and temperature                    |
| `orthrus/`                                           | drafter model, chain sampling, fused speculative decoder                            |
| `inference/`                                         | FP8 kernel, batched speculative engine, Jev-compatible server and benchmark         |
| `sdk/`                                               | `jeeves_sdk`, a drop-in replacement for Jev's Python SDK with the reasoning options |

## Limitations

- Knowledge questions trail Jev (MMLU 0.793 vs 0.900, MMLU-Pro 0.739 vs 0.840).
- Thinking is slow at the tail: 17 s at p90 with full chains. Use `max_think` and `nothink_threshold` when latency matters.
- About a third of full-length chains hit the 2,560-token cap without closing; the answer is still read out, and accuracy on those items is lower.
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

Model

- Qwen Team. [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), the base model. See also [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025.
- Yang, Kautz, Hatamizadeh. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). ICLR 2025.
- Yang, Wang, Zhang, Shen, Kim. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). NeurIPS 2024. Kernels from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
- Qiu et al. [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708). 2025.
- Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- Vinyals, Fortunato, Jaitly. [Pointer Networks](https://arxiv.org/abs/1506.03134). NeurIPS 2015.
- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked), the description of Jev's design that Kev and this project follow.

Training

- MiniMax. [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585). 2025. Introduces CISPO.
- Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). 2024. Group-relative advantages (GRPO).
- Guo, Pleiss, Sun, Weinberger. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). ICML 2017. Temperature scaling.
- Naeini, Cooper, Hauskrecht. [Obtaining Well Calibrated Probabilities Using Bayesian Binning](https://ojs.aaai.org/index.php/AAAI/article/view/9602). AAAI 2015. Expected calibration error.

Drafting and inference

- Leviathan, Kalman, Matias. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). ICML 2023.
- Chen et al. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318). 2023.
- Stern, Shazeer, Uszkoreit. [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115). NeurIPS 2018.
- Cai et al. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774). ICML 2024.
- Zhou et al. [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461). ICLR 2024.
- Hinton, Vinyals, Dean. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). 2015.
- Micikevicius et al. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433). 2022.

Evaluation

- Hendrycks et al. [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300). ICLR 2021.
- Wang et al. [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574). NeurIPS 2024.
- Dataset revisions for every training and evaluation source are pinned in `data/manifest.json`.
