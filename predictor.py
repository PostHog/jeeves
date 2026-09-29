from __future__ import annotations

import os
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from checkpoint import Checkpoint
from distributed import all_reduce_, setup_distributed
from loader.dataloader import Batch, Encoder, Example, collate_prompts, collate_rollouts, make_loader
from model import LoRAConfig, Qwen3_5ForCausalLM
from model.head import PointerHead


@dataclass
class PredictorConfig:
    model: str = "Qwen/Qwen3.5-9B"
    init: str | None = None
    lora_r: int = 16
    lora_alpha: float = 32.0
    head_dim: int = 256
    eval_batch_size: int = 16
    eval_think_limit: int = 0
    seed: int = 0
    max_think: int = 2560
    temperature: float = 0.0
    gen_batch_size: int = 64
    pad_multiple: int = 128
    num_workers: int = 2
    compile: bool = False

    @classmethod
    def from_checkpoint(cls, path: str, *, model: str | None = None, **overrides) -> PredictorConfig:
        checkpoint = Checkpoint.read(path)
        return cls(**(checkpoint.model_options(model) | {"init": str(checkpoint.path)} | overrides))


def stratified_subset(examples: list[Example], limit: int, seed: int) -> list[Example]:
    if not limit or limit >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = {}
    for ex in examples:
        buckets.setdefault(ex.source, []).append(ex)
    queues = [list(v) for v in buckets.values()]
    for q in queues:
        rng.shuffle(q)
    out: list[Example] = []
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop())
    return out


class Predictor:
    def __init__(self, cfg: PredictorConfig, device: str | None = None):
        self.cfg = cfg
        self.rank, self.world, self.local_rank = setup_distributed()
        self.device = torch.device(device) if device is not None else torch.device("cuda", self.local_rank)
        torch.manual_seed(cfg.seed + self.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.encoder = Encoder(cfg.model, pad_multiple=cfg.pad_multiple)
        self.m = self.encoder.markers
        self.model = Qwen3_5ForCausalLM.from_pretrained(cfg.model, device=self.device, lora=LoRAConfig(r=cfg.lora_r, alpha=cfg.lora_alpha))
        self.logit_bias = torch.zeros(self.model.lm_head.weight.shape[0], device=self.device)
        self.logit_bias[torch.tensor(self.encoder.banned_ids, device=self.device, dtype=torch.long)] = -1e4
        if cfg.compile:
            self.model.compile_layers()
        self.head = PointerHead(self.model.cfg.hidden_size, cfg.head_dim).to(self.device)
        self.sources: dict[str, int] = {}
        self.dtype = self.model.lm_head.weight.dtype
        if cfg.init:
            self.load(cfg.init)
        self.model.lora.refresh_cache(self.dtype)
        self.model.eval()
        self.head.eval()
        self.capture_lock = threading.Lock()

    def load(self, path: str) -> None:
        checkpoint = Checkpoint.read(path)
        checkpoint.load_weights(self.model, self.head, self.device)
        self.sources = dict(checkpoint.metadata.get("sources", {}))
        self.model.lora.refresh_cache(self.dtype)

    @contextmanager
    def evaluation(self, calibrated: bool = True):
        model_training, head_training = self.model.training, self.head.training
        self.model.eval()
        self.head.train(not calibrated)
        try:
            with torch.no_grad():
                yield
        finally:
            self.model.train(model_training)
            self.head.train(head_training)

    def scores(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        B, K = batch.opt_mask.shape
        flat = hidden.reshape(-1, hidden.shape[-1])
        decide = flat.index_select(0, batch.decide_index)
        options = flat.index_select(0, batch.opt_index.reshape(-1)).view(B, K, -1)
        logits = self.head(options, decide).float()
        return logits.masked_fill(~batch.opt_mask, float("-inf"))

    @staticmethod
    def head_nll(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        return -(batch.targets * F.log_softmax(logits, dim=-1).masked_fill(~batch.opt_mask, 0.0)).sum(-1)

    @torch.no_grad()
    def generate_chains(self, examples: list[Example], plain: bool = False) -> tuple[list[list[int]], list[list[float]]]:
        cfg = self.cfg
        chains: list[list[int]] = []
        lps: list[list[float]] = []
        for start in range(0, len(examples), cfg.gen_batch_size):
            chunk = examples[start:start + cfg.gen_batch_size]
            ids, mask = collate_prompts(chunk, self.m, plain=plain)
            ids, mask = ids.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)
            gen, lp = self.model.generate_graphed(ids, attention_mask=mask, max_new_tokens=cfg.max_think, temperature=cfg.temperature,
                                                  eos_token_id=self.encoder.think_end_id, return_logprobs=True,
                                                  capture_lock=self.capture_lock, logit_bias=self.logit_bias)
            lps.extend(lp.cpu().tolist())
            chains.extend(gen[:, ids.shape[1]:].cpu().tolist())
        return chains, lps

    def fit_temperature(self, examples: list[Example]) -> dict:
        with self.evaluation(calibrated=False):
            return self._fit_temperature(examples)

    def _fit_temperature(self, examples: list[Example]) -> dict:
        grid = torch.linspace(-1.5, 2.0, 351, device=self.device).exp()
        totals = torch.zeros(len(grid) + 1, device=self.device)
        loader = make_loader(examples[self.rank::self.world], self.cfg.eval_batch_size, self.m, shuffle=False, drop_last=False,
                             num_workers=self.cfg.num_workers)
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.scores(self.model.hidden_states(batch.input_ids, batch.attention_mask), batch)
            lp = F.log_softmax(logits[None] / grid[:, None, None], dim=-1).masked_fill(~batch.opt_mask[None], 0.0)
            totals[:-1] -= (batch.targets[None] * lp).sum((1, 2))
            totals[-1] += logits.shape[0]
        all_reduce_(totals, self.world)
        nll = totals[:-1] / totals[-1].clamp(min=1)
        best = int(nll.argmin())
        self.head.temperature = float(grid[best])
        one = int((grid - 1).abs().argmin())
        return {"temperature": round(self.head.temperature, 4), "dev_nll_before": round(float(nll[one]), 4),
                "dev_nll_after": round(float(nll[best]), 4), "rows": int(totals[-1])}

    def evaluate(self, examples: list[Example], think: bool = False) -> dict:
        with self.evaluation():
            return self._evaluate(examples, think)

    def _evaluate(self, examples: list[Example], think: bool) -> dict:
        cfg = self.cfg
        n_sources = max(len(self.sources), 1)
        totals = torch.zeros(3, device=self.device)
        per_source = torch.zeros(8, n_sources, device=self.device)
        shard = examples[self.rank::self.world]
        think_shard = stratified_subset(examples, cfg.eval_think_limit, cfg.seed)[self.rank::self.world] if think else []
        think_totals = torch.zeros(4, device=self.device)
        loader = make_loader(shard, cfg.eval_batch_size, self.m, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.scores(self.model.hidden_states(batch.input_ids, batch.attention_mask), batch)
            nll = self.head_nll(logits, batch)
            correct = (logits.argmax(-1) == batch.labels).float()
            max_p = F.softmax(logits, dim=-1).amax(-1)
            totals += torch.stack([correct.sum(), nll.sum(), correct.numel() + torch.zeros((), device=self.device)])
            per_source[0].index_add_(0, batch.source_ids, correct)
            per_source[1].index_add_(0, batch.source_ids, torch.ones_like(correct))
            per_source[4].index_add_(0, batch.source_ids, max_p)
            per_source[5].index_add_(0, batch.source_ids, (max_p >= 0.9).float())
        for start in range(0, len(think_shard), cfg.eval_batch_size):
            chunk = think_shard[start:start + cfg.eval_batch_size]
            generated, _ = self.generate_chains(chunk)
            batch = collate_rollouts(chunk, generated, list(range(len(chunk))), self.m).to(self.device)
            logits = self.scores(self.model.hidden_states(batch.input_ids, batch.attention_mask), batch)
            correct = (logits.argmax(-1) == batch.labels).float()
            max_p = F.softmax(logits, dim=-1).amax(-1)
            closed = batch.valid.float()
            think_totals += torch.stack([correct.sum(), closed.sum(), (correct * closed).sum(),
                                         correct.numel() + torch.zeros((), device=self.device)])
            per_source[2].index_add_(0, batch.source_ids, correct)
            per_source[3].index_add_(0, batch.source_ids, torch.ones_like(correct))
            per_source[6].index_add_(0, batch.source_ids, max_p)
            per_source[7].index_add_(0, batch.source_ids, (max_p >= 0.9).float())
        all_reduce_(totals, self.world)
        all_reduce_(per_source, self.world)
        all_reduce_(think_totals, self.world)
        t = totals.tolist()
        ps = per_source.tolist()
        names = {v: k for k, v in self.sources.items()}
        result = {"dev_acc": t[0] / max(1, t[2]), "dev_nll": t[1] / max(1, t[2]), "dev_n": int(t[2]),
                  "per_source": {names.get(i, str(i)): round(ps[0][i] / ps[1][i], 4) for i in range(n_sources) if ps[1][i] > 0}}
        unk, ctl = self.sources.get("unknowable"), self.sources.get("unknowable_control")
        if unk is not None and ps[1][unk] > 0:
            result["unknowable"] = {"n": int(ps[1][unk]), "mean_max_p": round(ps[4][unk] / ps[1][unk], 4),
                                    "share_at_0_9": round(ps[5][unk] / ps[1][unk], 4)}
            if ctl is not None and ps[1][ctl] > 0:
                result["unknowable"].update(control_acc=round(ps[0][ctl] / ps[1][ctl], 4),
                                            control_mean_max_p=round(ps[4][ctl] / ps[1][ctl], 4),
                                            control_share_at_0_9=round(ps[5][ctl] / ps[1][ctl], 4))
            if think and ps[3][unk] > 0:
                result["unknowable"].update(think_mean_max_p=round(ps[6][unk] / ps[3][unk], 4),
                                            think_share_at_0_9=round(ps[7][unk] / ps[3][unk], 4))
        if think:
            tt = think_totals.tolist()
            result.update(dev_think_acc=tt[0] / max(1, tt[3]), dev_think_closed=tt[1] / max(1, tt[3]),
                          dev_think_acc_closed=tt[2] / max(1, tt[1]), dev_think_n=int(tt[3]),
                          per_source_think={names.get(i, str(i)): round(ps[2][i] / ps[3][i], 4) for i in range(n_sources) if ps[3][i] > 0})
        return result
