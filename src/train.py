import os
import csv
import math
import time
import argparse
import numpy as np
import torch
from contextlib import nullcontext
from decoder_only_transformer import GPT, ModelConfig

CSV_FIELDS = [
    "step", "tokens_seen", "train_loss", "val_loss_total",
    "val_loss_acts", "val_loss_high", "val_loss_supreme", "val_loss_tribunal",
    "lr", "grad_norm", "tokens_per_sec", "mfu", "elapsed_hours",
]

NATURAL_MIX = {"acts": 0.040, "high": 0.427, "supreme": 0.072, "tribunal": 0.461}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MemmapDataLoader:
    def __init__(self, data_dir, block_size, val_tokens=5_000_000, weights=None):
        self.block_size = block_size
        self.val_tokens = val_tokens
        self.data = {}

        weights = dict(weights or NATURAL_MIX)
        for src in list(weights):
            path = os.path.join(data_dir, f"{src}.bin")
            if not os.path.exists(path):
                print(f"warning: {path} missing, dropped from mixture")
                del weights[src]
                continue
            self.data[src] = np.memmap(path, dtype=np.uint16, mode="r")

        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.sources = list(self.weights)
        self.probs = [self.weights[s] for s in self.sources]
        if not self.sources:
            raise FileNotFoundError(f"no .bin files found in {data_dir}")

        for s in self.sources:
            n = len(self.data[s])
            print(f"  {s:9s} {n:,} tokens  (train {n - val_tokens:,} / val {val_tokens:,})")

        self.eval_rng = torch.Generator()

    def _slice(self, arr, idx):
        x = torch.from_numpy(arr[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(arr[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y

    def _rand_idx(self, arr, split, gen=None):
        train_end = len(arr) - self.val_tokens
        if split == "train":
            hi = train_end - self.block_size - 1
            return torch.randint(0, hi, (1,)).item()
        lo, hi = train_end, len(arr) - self.block_size - 1
        return torch.randint(lo, hi, (1,), generator=gen).item()

    def get_batch(self, split, batch_size, device, source=None):
        xs, ys = [], []
        gen = self.eval_rng if split == "val" else None
        srcs = ([source] * batch_size if source
                else np.random.choice(self.sources, size=batch_size, p=self.probs))

        for s in srcs:
            arr = self.data[s]
            x, y = self._slice(arr, self._rand_idx(arr, split, gen))
            xs.append(x); ys.append(y)

        x = torch.stack(xs).pin_memory().to(device, non_blocking=True)
        y = torch.stack(ys).pin_memory().to(device, non_blocking=True)
        return x, y


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    r = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, dl, batch_size, train_iters, val_iters, ctx, device):
    model.eval()
    out = {}

    losses = torch.zeros(train_iters)
    for k in range(train_iters):
        X, Y = dl.get_batch("train", batch_size, device)
        with ctx:
            _, loss = model(X, targets=Y)
        losses[k] = loss.item()
    out["train"] = losses.mean().item()

    dl.eval_rng.manual_seed(1337)          # same windows every eval
    out["val_sources"] = {}
    total = 0.0
    for src in dl.sources:
        ls = torch.zeros(val_iters)
        for k in range(val_iters):
            X, Y = dl.get_batch("val", batch_size, device, source=src)
            with ctx:
                _, loss = model(X, targets=Y)
            ls[k] = loss.item()
        m = ls.mean().item()
        out["val_sources"][src] = m
        total += m * dl.weights[src]
    out["val_total"] = total

    model.train()
    return out


def parse_mixture(s):
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--data_dir", default=os.path.join(ROOT, "training_data/tokens"))
    p.add_argument("--out_dir",  default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--mixture", default="", help='e.g. "acts=0.2,supreme=0.2,high=0.3,tribunal=0.3"')

    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--max_lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_train_iters", type=int, default=40)
    p.add_argument("--eval_val_iters", type=int, default=20)
    p.add_argument("--checkpoint_interval", type=int, default=2000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--peak_tflops", type=float, default=150.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else nullcontext())

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_csv = os.path.join(run_dir, "train_log.csv")

    print("data sources:")
    dl = MemmapDataLoader(args.data_dir, args.block_size,
                          weights=parse_mixture(args.mixture))
    print(f"mixture: {dl.weights}\n")

    config = ModelConfig(block_size=args.block_size, n_layer=args.n_layer,
                         n_head=args.n_head, d_model=args.d_model,
                         dropout=args.dropout)
    model = GPT(config)
    print(f"model: {sum(q.numel() for q in model.parameters()):,} params")

    step, best_val, tokens_seen = 0, float("inf"), 0

    # load state BEFORE compiling
    if args.resume:
        ckpt_path = os.path.join(run_dir, "ckpt_latest.pt")
        if os.path.exists(ckpt_path):
            print(f"resuming from {ckpt_path}")
            ck = torch.load(ckpt_path, map_location="cpu", weights_only = False)
            sd = ck["model"]
            for k in list(sd):
                if k.startswith("_orig_mod."):
                    sd[k[len("_orig_mod."):]] = sd.pop(k)
            model.load_state_dict(sd)
            step = ck["step"]
            best_val = ck["best_val_loss"]
            tokens_seen = ck.get("tokens_seen", 0)
            torch.set_rng_state(ck["torch_rng"])
            np.random.set_state(ck["numpy_rng"])
        else:
            print(f"no checkpoint at {ckpt_path}, starting fresh")

    model.to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.max_lr, device)
    if args.resume and os.path.exists(os.path.join(run_dir, "ckpt_latest.pt")):
        optimizer.load_state_dict(ck["optimizer"])

    raw_model = model
    if args.compile:
        print("compiling...")
        model = torch.compile(model)

    n_params = sum(q.numel() for q in raw_model.parameters())
    flops_per_token = 6 * n_params
    tokens_per_step = args.batch_size * args.grad_accum_steps * args.block_size
    print(f"tokens/step: {tokens_per_step:,}  -> "
          f"{tokens_per_step * args.max_steps / 1e9:.2f}B over {args.max_steps} steps\n")

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    last = {"grad_norm": 0.0, "tokens_per_sec": 0.0, "mfu": 0.0}
    t_start = time.time()
    t0 = time.time()
    model.train()

    while step <= args.max_steps:
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % args.eval_interval == 0 or step == args.max_steps:
            L = estimate_loss(model, dl, args.batch_size, args.eval_train_iters,
                              args.eval_val_iters, ctx, device)
            row = {f: "" for f in CSV_FIELDS}
            row.update({
                "step": step, "tokens_seen": tokens_seen,
                "train_loss": round(L["train"], 5),
                "val_loss_total": round(L["val_total"], 5),
                "lr": f"{lr:.3e}",
                "elapsed_hours": round((time.time() - t_start) / 3600, 3),
                **{k: round(v, 5) for k, v in last.items()},
            })
            for s, v in L["val_sources"].items():
                row[f"val_loss_{s}"] = round(v, 5)

            with open(log_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

            src_str = "  ".join(f"{s}:{v:.3f}" for s, v in L["val_sources"].items())
            print(f"EVAL step {step:6d} | train {L['train']:.4f} | "
                  f"val {L['val_total']:.4f} | {src_str}")

            ck = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
                "step": step, "best_val_loss": best_val,
                "tokens_seen": tokens_seen,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            }
            torch.save(ck, os.path.join(run_dir, "ckpt_latest.pt"))
            if L["val_total"] < best_val:
                best_val = L["val_total"]
                ck["best_val_loss"] = best_val
                torch.save(ck, os.path.join(run_dir, "ckpt_best.pt"))
            if step > 0 and step % args.checkpoint_interval == 0:
                torch.save(ck, os.path.join(run_dir, f"ckpt_step_{step}.pt"))

        if step == args.max_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum_steps):
            X, Y = dl.get_batch("train", args.batch_size, device)
            with ctx:
                _, loss = model(X, targets=Y)
                loss = loss / args.grad_accum_steps
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        t0 = time.time()

        tokens_seen += tokens_per_step
        tps = tokens_per_step / dt
        last = {"grad_norm": float(norm), "tokens_per_sec": tps,
                "mfu": flops_per_token * tps / (args.peak_tflops * 1e12)}

        if step % args.log_interval == 0:
            print(f"step {step:6d} | loss {loss.item()*args.grad_accum_steps:.4f} | "
                  f"norm {float(norm):.2f} | lr {lr:.2e} | {dt:.2f}s | "
                  f"{tps:,.0f} tok/s | MFU {last['mfu']*100:.1f}%")

        step += 1


if __name__ == "__main__":
    train()