from __future__ import annotations

import json
import math
import queue
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file

from checkpoint import Checkpoint
from distributed import all_reduce_, all_reduce_grads, setup_distributed
from loader.dataloader import Encoder, Example, collate_rollouts, collate_sft, load_examples, make_loader, split_chain
from metrics import Metrics
from predictor import Predictor, PredictorConfig


@dataclass
class Config(PredictorConfig):
    stage: str = "sft"
    run_dir: str = "runs/run"
    data_dir: str = "data"
    resume: str | None = None
    lr: float = 5e-5
    head_lr: float = 1e-4
    warmup: int = 20
    sft_epochs: int = 2
    cispo_epochs: int = 4
    stop_step: int = 402
    batch_size: int = 8
    grad_clip: float = 1.0
    max_len: int = 4096
    max_seq_len: int = 8192
    train_limit: int = 0
    dev_limit: int = 0
    eval_every: int = 50
    eval_think: bool = True
    eval_think_limit: int = 256
    log_every: int = 10
    gradient_checkpointing: bool = True
    prompts_per_step: int = 8
    group: int = 8
    temperature: float = 1.0
    eps_high: float = 0.2
    rollout_ce_weight: float = 0.5
    anchor_ce_weight: float = 0.5
    anchor_batch: int = 8
    logprob_chunk: int = 128
    micro_groups: int = 2
    length_hinge: int = 2048
    length_hinge_end: int = 1024
    length_penalty_cap: float = 0.1
    compile: bool = True
    ckpt_minutes: float = 45.0
    log_rollouts: int = 3
    log_chain_chars: int = 1200
    sft_think_frac: float = 0.5
    chain_cache: str = "data/chains"


_TRITON_LOCK = threading.RLock()
_TRITON_PATCHED = False


def _locked_run(fn):
    def run(self, *args, **kwargs):
        with _TRITON_LOCK:
            return fn(self, *args, **kwargs)
    return run


def enable_threadsafe_triton() -> None:
    global _TRITON_PATCHED
    import triton.runtime.autotuner
    import triton.runtime.jit

    with _TRITON_LOCK:
        if not _TRITON_PATCHED:
            triton.runtime.autotuner.Autotuner.run = _locked_run(triton.runtime.autotuner.Autotuner.run)
            triton.runtime.jit.JITFunction.run = _locked_run(triton.runtime.jit.JITFunction.run)
            _TRITON_PATCHED = True


def lr_lambda(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


class Checkpointer:
    def __init__(self, run_dir: Path, enabled: bool):
        self.run_dir = run_dir
        self.enabled = enabled
        self.stream = torch.cuda.Stream() if enabled else None
        self.pinned: dict[str, torch.Tensor] = {}
        self.lock = threading.Lock()
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True) if enabled else None
        if self.thread:
            self.thread.start()
        self.last = time.time()

    def _buffer(self, name: str, t: torch.Tensor) -> torch.Tensor:
        buf = self.pinned.get(name)
        if buf is None or buf.shape != t.shape or buf.dtype != t.dtype:
            buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            self.pinned[name] = buf
        return buf

    def submit(self, tensors: dict[str, torch.Tensor], meta: dict) -> None:
        if not self.enabled:
            return
        self.lock.acquire()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for name, t in tensors.items():
                self._buffer(name, t).copy_(t, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.jobs.put((event, list(tensors.keys()), meta))
        self.last = time.time()

    def _clear(self, d: Path) -> None:
        if d.exists():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()

    def _worker(self) -> None:
        while True:
            event, keys, meta = self.jobs.get()
            event.synchronize()
            tmp, final, old = self.run_dir / "ckpt.tmp", self.run_dir / "ckpt", self.run_dir / "ckpt.old"
            tmp.mkdir(parents=True, exist_ok=True)
            state = {k: self.pinned[k] for k in keys}
            save_file({k: v.contiguous() for k, v in state.items() if k.startswith("lora.")}, str(tmp / "lora.safetensors"))
            torch.save({k[5:]: v.clone() for k, v in state.items() if k.startswith("head.")}, tmp / "head.pt")
            torch.save({k[4:]: v.clone() for k, v in state.items() if k.startswith("opt.")}, tmp / "opt.pt")
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            self.lock.release()
            if final.exists():
                self._clear(old)
                final.rename(old)
            tmp.rename(final)
            self._clear(old)

    def due(self, minutes: float) -> bool:
        return self.enabled and minutes > 0 and time.time() - self.last >= minutes * 60

    def wait(self) -> None:
        if self.enabled:
            with self.lock:
                pass


class RolloutWorker:
    def __init__(self, fn):
        enable_threadsafe_triton()
        self.fn = fn
        self.stream = torch.cuda.Stream()
        self.jobs: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        torch.set_grad_enabled(False)
        while True:
            job = self.jobs.get()
            if job is None:
                return
            tag, args = job
            try:
                with torch.cuda.stream(self.stream):
                    out = self.fn(*args)
                    self.stream.synchronize()
                torch.cuda.empty_cache()
                self.results.put((tag, out, None))
            except BaseException as e:
                self.results.put((tag, None, e))

    def submit(self, tag, *args) -> None:
        self.jobs.put((tag, args))

    def get(self):
        tag, out, err = self.results.get()
        if err is not None:
            raise err
        return tag, out

    def stop(self) -> None:
        self.jobs.put(None)


class Trainer(Predictor):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.model.gradient_checkpointing_enable(cfg.gradient_checkpointing)
        if cfg.stage in ("sft", "cispo"):
            self.head.temperature = 1.0
            self.head.train()
        self.lora_params = list(self.model.lora.parameters())
        self.head_params = list(self.head.parameters())
        self.params = self.lora_params + self.head_params
        self.opt = torch.optim.AdamW([{"params": self.lora_params, "lr": cfg.lr}, {"params": self.head_params, "lr": cfg.head_lr}],
                                     weight_decay=0.0, betas=(0.9, 0.95), fused=True)
        self.sched = None
        self.step = 0
        self.run_dir = Path(cfg.run_dir)
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        self.log_file = open(self.run_dir / "log.jsonl", "a") if self.rank == 0 else None
        self.last_eval: dict | None = None
        self.ckpt = Checkpointer(self.run_dir, enabled=self.rank == 0 and cfg.ckpt_minutes > 0)
        self.resumed_meta: dict | None = None
        if cfg.resume:
            self.load_state(cfg.resume)

    def log(self, **kv) -> None:
        if self.rank != 0:
            return
        kv = {"step": self.step, "time": time.time(), **kv}
        print(json.dumps(kv), flush=True)
        self.log_file.write(json.dumps(kv) + "\n")
        self.log_file.flush()

    def save(self, metrics: dict | None = None, name: str = "final") -> None:
        if self.rank != 0:
            return
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        self.model.lora.save(str(d / "lora.safetensors"))
        torch.save(self.head.state_dict(), d / "head.pt")
        (d / "state.json").write_text(json.dumps({"step": self.step, "stage": self.cfg.stage, "sources": self.sources,
                                                  "temperature": self.head.temperature, "metrics": metrics,
                                                  "config": asdict(self.cfg)}, indent=2) + "\n")
        if name != "final":
            return
        latest = self.run_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(d.name)

    def state_tensors(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in self.model.lora.named_parameters():
            out["lora." + k] = v.detach()
        for k, v in self.head.state_dict().items():
            out["head." + k] = v.detach()
        for pid, st in self.opt.state_dict()["state"].items():
            for name, v in st.items():
                if isinstance(v, torch.Tensor):
                    out[f"opt.{pid}.{name}"] = v.detach()
        return out

    def save_state(self, extra: dict) -> None:
        if self.rank != 0:
            return
        meta = {"step": self.step, "stage": self.cfg.stage, "sources": self.sources, "config": asdict(self.cfg),
                "torch_rng": torch.get_rng_state().tolist(), **extra}
        self.ckpt.submit(self.state_tensors(), meta)
        self.log(event="checkpoint", **extra)

    def load_state(self, path: str) -> None:
        checkpoint = Checkpoint.read(path)
        p, meta = checkpoint.path, checkpoint.metadata
        checkpoint.load_weights(self.model, self.head, self.device)
        opt_tensors = torch.load(p / "opt.pt", map_location=self.device)
        sd = self.opt.state_dict()
        state: dict = {}
        for k, v in opt_tensors.items():
            pid, name = k.split(".", 1)
            state.setdefault(int(pid), {})[name] = v
        sd["state"] = state
        self.opt.load_state_dict(sd)
        self.step = meta["step"]
        self.sources = dict(meta.get("sources", self.sources))
        self.model.lora.refresh_cache(self.dtype)
        self.resumed_meta = meta
        self.log(event="resumed", from_dir=str(p), step=self.step)

    def optimizer_step(self) -> torch.Tensor:
        all_reduce_grads(self.params, self.world)
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        self.model.lora.refresh_cache(self.dtype)
        self.step += 1
        return norm

    def make_schedule(self, total: int) -> None:
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(self.cfg.warmup, total))
        for _ in range(self.step):
            self.sched.step()

    def checkpoint_due(self) -> bool:
        if self.cfg.ckpt_minutes <= 0:
            return False
        if self.world == 1:
            return self.ckpt.due(self.cfg.ckpt_minutes)
        flag = torch.tensor([float(self.rank == 0 and self.ckpt.due(self.cfg.ckpt_minutes))], device=self.device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def chain_cache_path(self) -> Path:
        tag = f"{self.cfg.model.replace('/', '_')}_{Encoder.FORMAT}_t{self.cfg.temperature:g}_k{self.cfg.max_think}"
        return Path(self.cfg.chain_cache) / f"{tag}.jsonl"

    def attach_chains(self, train: list[Example]) -> dict:
        cfg = self.cfg
        n = int(round(cfg.sft_think_frac * len(train)))
        if n == 0:
            return {"with_chains": 0}
        selected = sorted(random.Random(f"{cfg.seed}:sft-think").sample(range(len(train)), n))
        path = self.chain_cache_path()
        cached: dict[str, list[int]] = {}
        if path.exists():
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        cached[row["id"]] = row["chain"]
        key = lambda ex: f"{ex.record_id}/{ex.question_id}"
        missing = [i for i in selected if key(train[i]) not in cached]
        mine = missing[self.rank::self.world]
        local = list(zip(mine, self.generate_chains([train[i] for i in mine], plain=True)[0])) if mine else []
        if self.world > 1:
            gathered: list = [None] * self.world
            dist.all_gather_object(gathered, local)
            local = [item for part in gathered for item in part]
        if self.rank == 0 and local:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                for i, chain in local:
                    fh.write(json.dumps({"id": key(train[i]), "chain": chain}) + "\n")
        for i, chain in local:
            cached[key(train[i])] = chain
        closed = lengths = 0
        for i in selected:
            chain = cached[key(train[i])]
            train[i].think = chain
            c, ok = split_chain(chain, self.m)
            closed += int(ok)
            lengths += len(c)
        return {"with_chains": len(selected), "generated": len(local), "from_cache": len(selected) - len(local),
                "cache": str(path), "chains_closed": closed / max(1, len(selected)), "chain_mean_len": lengths / max(1, len(selected))}

    def train_sft(self, train: list[Example], dev: list[Example]) -> None:
        cfg = self.cfg
        self.model.eval()
        self.log(event="chains", **self.attach_chains(train))
        loader = make_loader(train, cfg.batch_size, self.m, shuffle=True, rank=self.rank, world_size=self.world, seed=cfg.seed,
                             num_workers=cfg.num_workers)
        total = cfg.sft_epochs * len(loader)
        self.make_schedule(total)
        if self.step == 0:
            self.last_eval = self.evaluate(dev)
            self.log(event="eval", **self.last_eval)
        acc = Metrics(("loss", "acc", "grad_norm", "tokens"), self.device)
        t0 = time.time()
        self.model.train()
        start_epoch, skip = divmod(self.step, len(loader))
        for epoch in range(start_epoch, cfg.sft_epochs):
            if self.world > 1:
                loader.sampler.set_epoch(epoch)
            for b_idx, batch in enumerate(loader):
                if epoch == start_epoch and b_idx < skip:
                    continue
                batch = batch.to(self.device)
                logits = self.scores(self.model.hidden_states(batch.input_ids, batch.attention_mask), batch)
                loss = self.head_nll(logits, batch).mean()
                loss.backward()
                norm = self.optimizer_step()
                acc.add(loss=loss, acc=(logits.argmax(-1) == batch.labels).float().mean(),
                        grad_norm=norm, tokens=batch.attention_mask.sum().float())
                if self.step % cfg.log_every == 0:
                    vals = acc.means(self.world)
                    tokens = vals.pop("tokens") * acc.count * self.world
                    self.log(event="train", epoch=epoch, **vals,
                             tokens_per_s=tokens / (time.time() - t0), lr=self.sched.get_last_lr()[0])
                    acc.reset()
                    t0 = time.time()
                if cfg.eval_every and self.step % cfg.eval_every == 0:
                    self.last_eval = self.evaluate(dev)
                    self.log(event="eval", **self.last_eval)
                if self.checkpoint_due():
                    self.save_state({"epoch": epoch})
        if not cfg.eval_every or self.step % cfg.eval_every:
            self.last_eval = self.evaluate(dev)
            self.log(event="eval", **self.last_eval)
        self.save(self.last_eval)

    def hinge_at(self, step: int, total: int) -> float:
        start, end = float(self.cfg.length_hinge), float(self.cfg.length_hinge_end)
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))

    def anchor_loss(self, anchors: list[Example]) -> torch.Tensor:
        batch = collate_sft(anchors, self.m).pin_memory().to(self.device, non_blocking=True)
        hidden = self.model.hidden_states(batch.input_ids, batch.attention_mask)
        return self.head_nll(self.scores(hidden, batch), batch).mean()

    def prompt_batches(self, shard: list[Example], per_step: int, start_epoch: int = 0, skip: int = 0):
        epoch = start_epoch
        while True:
            order = list(range(len(shard)))
            random.Random(f"{self.cfg.seed}:{self.rank}:{epoch}").shuffle(order)
            for b, start in enumerate(range(0, len(order) - per_step + 1, per_step)):
                if epoch == start_epoch and b < skip:
                    continue
                yield epoch, b, [shard[i] for i in order[start:start + per_step]]
            epoch += 1

    def train_cispo(self, pool: list[Example], anchors: list[Example], dev: list[Example]) -> None:
        cfg = self.cfg
        shard = pool[self.rank::self.world]
        total = cfg.cispo_epochs * (len(shard) // cfg.prompts_per_step)
        end = min(total, cfg.stop_step) if cfg.stop_step else total
        self.make_schedule(total)
        start_epoch, skip = 0, 0
        if self.resumed_meta is not None:
            start_epoch, skip = self.resumed_meta.get("epoch", 0), self.resumed_meta.get("batch", 0)
        batches = self.prompt_batches(shard, cfg.prompts_per_step, start_epoch, skip)
        self.log(event="plan", total_steps=total, stop_step=end, epochs=cfg.cispo_epochs, hinge_start=cfg.length_hinge,
                 hinge_end=cfg.length_hinge_end, penalty_cap=cfg.length_penalty_cap, anchor_ce_weight=cfg.anchor_ce_weight,
                 anchor_batch=cfg.anchor_batch, anchor_pool=len(anchors), rl_prompts=len(pool), start_epoch=start_epoch, start_batch=skip)
        if self.step == 0:
            self.last_eval = self.evaluate(dev, think=cfg.eval_think)
            self.log(event="eval", **self.last_eval)
        acc = Metrics(("loss", "pg", "ce_rollout", "ce_anchor", "reward", "rollout_acc", "grad_norm",
                       "tokens", "mean_think_len", "closed_frac", "penalized_frac"), self.device)
        t0 = time.time()
        P, G = cfg.prompts_per_step, cfg.group
        n_rows = float(P * G)
        anchor_rng = random.Random(f"{cfg.seed}:anchor:{self.rank}")
        group_ids = [i for i in range(P) for _ in range(G)]
        rows_per_micro = cfg.micro_groups * G
        self.model.eval()
        worker = RolloutWorker(self.generate_chains)

        def submit_next() -> None:
            epoch, b, prompts = next(batches)
            expanded = [ex for ex in prompts for _ in range(G)]
            worker.submit((epoch, b, expanded), expanded)

        submit_next()
        while self.step < end:
            (epoch, b, expanded), (generated, lps) = worker.get()
            deferred = False
            if end - self.step > 1:
                if (cfg.eval_every and (self.step + 1) % cfg.eval_every == 0) or self.ckpt.due(cfg.ckpt_minutes):
                    deferred = True
                else:
                    submit_next()
            hinge = self.hinge_at(self.step, total)
            cpu_micro = [collate_rollouts(expanded[s:s + rows_per_micro], generated[s:s + rows_per_micro], group_ids[s:s + rows_per_micro],
                                          self.m, lps[s:s + rows_per_micro]) for s in range(0, P * G, rows_per_micro)]
            mask_total = max(1.0, float(sum(int(mb.chain_mask[:, 1:].sum()) for mb in cpu_micro)))
            batch_tokens = float(sum(int(mb.attention_mask.sum()) for mb in cpu_micro))
            micro = [mb.pin_memory().to(self.device, non_blocking=True) for mb in cpu_micro]
            stats = Metrics(("pg", "ce_rollout", "reward", "correct", "think_len", "closed", "penalized"), self.device)
            preds, probs = [], []
            for mb in micro:
                with self.capture_lock:
                    think_len = (mb.attention_mask.sum(1) - mb.think_start - mb.suffix_len).float()
                    penalty = (torch.relu(think_len - hinge) / cfg.length_hinge).clamp(max=cfg.length_penalty_cap)
                    hidden = self.model.hidden_states(mb.input_ids, mb.attention_mask)
                    logits = self.scores(hidden, mb)
                    nll = self.head_nll(logits, mb)
                    with torch.no_grad():
                        p_correct = F.softmax(logits, dim=-1).gather(1, mb.labels[:, None]).squeeze(1)
                        reward = p_correct * (1 - penalty)
                        gids = mb.group_ids - mb.group_ids.min()
                        gsum = torch.zeros(cfg.micro_groups, device=self.device).index_add_(0, gids, reward)
                        gcnt = torch.zeros(cfg.micro_groups, device=self.device).index_add_(0, gids, torch.ones_like(reward))
                        adv = reward - (gsum / gcnt.clamp(min=1))[gids]
                        preds.append(logits.argmax(-1))
                        probs.append(p_correct)
                    logp = self.model.token_logprobs(hidden[:, :-1], mb.input_ids[:, 1:], cfg.logprob_chunk, cfg.temperature, self.logit_bias)
                    mask = mb.chain_mask[:, 1:].float()
                    weight = torch.exp(logp - mb.old_logp[:, 1:]).clamp(max=1 + cfg.eps_high).detach()
                    pg = -(weight * adv[:, None] * logp * mask).sum() / mask_total
                    ce_roll = nll.sum() / n_rows
                    (pg + cfg.rollout_ce_weight * ce_roll).backward()
                    torch.cuda.current_stream().synchronize()
                stats.add(pg=pg, ce_rollout=ce_roll, reward=reward.sum(),
                          correct=(logits.argmax(-1) == mb.labels).float().sum(), think_len=think_len.sum(),
                          closed=mb.valid.float().sum(), penalized=(penalty > 0).float().sum())
            with self.capture_lock:
                ce_anchor = self.anchor_loss(anchor_rng.sample(anchors, min(cfg.anchor_batch, len(anchors))))
                (cfg.anchor_ce_weight * ce_anchor).backward()
                norm = self.optimizer_step()
                torch.cuda.current_stream().synchronize()
            acc.add(loss=stats["pg"] + cfg.rollout_ce_weight * stats["ce_rollout"] + cfg.anchor_ce_weight * ce_anchor.detach(),
                    pg=stats["pg"], ce_rollout=stats["ce_rollout"], ce_anchor=ce_anchor, reward=stats["reward"] / n_rows,
                    rollout_acc=stats["correct"] / n_rows, grad_norm=norm, tokens=batch_tokens,
                    mean_think_len=stats["think_len"] / n_rows, closed_frac=stats["closed"] / n_rows,
                    penalized_frac=stats["penalized"] / n_rows)
            if self.step % cfg.log_every == 0:
                vals = acc.means(self.world)
                tokens = vals.pop("tokens") * acc.count * self.world
                self.log(event="train", epoch=epoch, hinge=round(hinge, 1), **vals,
                         tokens_per_s=tokens / (time.time() - t0), lr=self.sched.get_last_lr()[0])
                acc.reset()
                t0 = time.time()
            if cfg.eval_every and self.step % cfg.eval_every == 0:
                self.log_rollout_samples(expanded, generated, torch.cat(preds), torch.cat(probs))
                self.last_eval = self.evaluate(dev, think=cfg.eval_think)
                self.log(event="eval", **self.last_eval)
                self.save(self.last_eval, name=f"evals/step_{self.step}")
            if self.checkpoint_due():
                self.save_state({"epoch": epoch, "batch": b + 1})
                self.ckpt.wait()
            if deferred:
                submit_next()
        worker.stop()
        if not cfg.eval_every or self.step % cfg.eval_every:
            self.last_eval = self.evaluate(dev, think=cfg.eval_think)
            self.log(event="eval", **self.last_eval)
        self.last_eval["calibration"] = self.fit_temperature(dev)
        self.log(event="calibration", **self.last_eval["calibration"])
        self.save(self.last_eval)

    def log_rollout_samples(self, examples: list[Example], generated: list[list[int]], preds: torch.Tensor, probs: torch.Tensor) -> None:
        if self.rank != 0 or not self.cfg.log_rollouts:
            return
        preds, probs = preds.tolist(), probs.tolist()
        samples = []
        for i in range(0, min(len(examples), self.cfg.log_rollouts * self.cfg.group), self.cfg.group):
            ex, gen = examples[i], generated[i]
            closed = self.encoder.think_end_id in gen
            chain = gen[:gen.index(self.encoder.think_end_id)] if closed else gen
            text = self.encoder.tok.decode(chain)
            options = [o.strip().split("<|box_end|>")[0] for o in self.encoder.tok.decode(ex.remainder).split("<|box_start|>")[1:]]
            samples.append({"source": ex.source, "id": f"{ex.record_id}/{ex.question_id}",
                            "prompt_tail": self.encoder.tok.decode(ex.prompt[-160:]),
                            "chain": text[:self.cfg.log_chain_chars] + (" ...[truncated]" if len(text) > self.cfg.log_chain_chars else ""),
                            "chain_tokens": len(chain), "closed": closed,
                            "label": options[ex.label] if ex.label < len(options) else ex.label,
                            "pred": options[preds[i]] if preds[i] < len(options) else preds[i],
                            "p_correct": round(probs[i], 4)})
        self.log(event="rollouts", samples=samples)


def run(cfg: Config) -> None:
    trainer = Trainer(cfg)
    enc, src = trainer.encoder, trainer.sources
    data = Path(cfg.data_dir)
    dev, d_drop = load_examples(str(data / "dev.jsonl"), enc, src, cfg.max_len, limit=cfg.dev_limit, seed=cfg.seed)
    if cfg.stage == "sft":
        train, t_drop = load_examples(str(data / "train.jsonl"), enc, src, cfg.max_len, limit=cfg.train_limit, seed=cfg.seed)
        trainer.log(event="data", train=len(train), train_dropped=t_drop, dev=len(dev), dev_dropped=d_drop, sources=src)
        trainer.train_sft(train, dev)
    elif cfg.stage == "cispo":
        pool, p_drop = load_examples(str(data / "rl.jsonl"), enc, src, cfg.max_seq_len, extra_len=cfg.max_think, limit=cfg.train_limit, seed=cfg.seed)
        anchors, a_drop = load_examples(str(data / "train.jsonl"), enc, src, cfg.max_seq_len, extra_len=cfg.max_think, seed=cfg.seed)
        trainer.log(event="data", rl=len(pool), rl_dropped=p_drop, anchors=len(anchors), anchors_dropped=a_drop, dev=len(dev), sources=src)
        trainer.train_cispo(pool, anchors, dev)
    else:
        raise ValueError(f"unknown stage {cfg.stage}")
    if trainer.world > 1:
        dist.barrier()
        dist.destroy_process_group()
