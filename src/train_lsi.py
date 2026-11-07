import os, csv, json, math, time, argparse

import numpy as np
import torch
from tokenizers import Tokenizer

from lsi_data import LSIDocs, load_split, batches, TOKENIZER, MODES, N_LABELS
from lsi_model import StatuteClassifier
from lsi_metrics import scores, best_global, best_per_label
from cjpe_model import POOLINGS
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FIELDS = ["epoch", "step", "train_loss", "macro_05", "macro_global",
              "thr_global", "macro_perlabel", "micro", "subset", "lr",
              "elapsed_min"]


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


@torch.no_grad()
def predict(model, ds, batch_size, device, ctx):
    model.eval()
    P, G = [], []
    for chunk, (ids, msk, y) in batches(ds, batch_size, device, shuffle=False):
        with ctx:
            logits, _ = model(ids, msk)
        P.append(torch.sigmoid(logits.float()).cpu())
        G.append(y.cpu())
    model.train()
    return torch.cat(P).numpy(), torch.cat(G).numpy().astype(int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--mode", default="head", choices=MODES)
    p.add_argument("--pooling", default="mean", choices=POOLINGS)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--pos_weight", default="none",
                   choices=["none", "inv", "capped"])
    p.add_argument("--finetune", default="full",
                   choices=["full", "lorma", "top_k", "head_only"])
    p.add_argument("--lorma_r", type=int, default=8)
    p.add_argument("--lorma_alpha", type=float, default=16.0)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--max_lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=3e-6)
    p.add_argument("--warmup_frac", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1899)
    p.add_argument("--limit_train", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    tok = Tokenizer.from_file(args.tokenizer)
    train_docs = load_split("train")
    if args.limit_train:
        train_docs = train_docs[:args.limit_train]
    dev_docs = load_split("dev")

    t0 = time.time()
    tr = LSIDocs(train_docs, tok, mode=args.mode, max_len=args.max_len)
    va = LSIDocs(dev_docs, tok, mode=args.mode, max_len=args.max_len)
    print(f"tokenised in {time.time()-t0:.0f}s | train {len(tr)} | "
          f"dev {len(va)} | mode {args.mode} | pooling {args.pooling}")

    pos = np.zeros(N_LABELS)
    for it in tr.items:
        for v in it["labels"]:
            pos[v] += 1
    rate = pos / max(1, len(tr))
    print(f"label positive rate: min {rate.min():.4f} median "
          f"{np.median(rate):.4f} max {rate.max():.4f}")

    pw = None
    if args.pos_weight != "none":
        w = (1 - rate) / np.clip(rate, 1e-6, None)
        if args.pos_weight == "capped":
            w = np.clip(w, 1.0, 20.0)
        pw = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"pos_weight: min {w.min():.2f} max {w.max():.2f}")

    base, arch, d_model = load_base(args.checkpoint, device)
    model = StatuteClassifier(base, arch, d_model, dropout=args.dropout,
                              pooling=args.pooling)

    if args.finetune == "lorma":
        from lorma import apply_lorma
        apply_lorma(model.base, r=args.lorma_r, alpha=args.lorma_alpha,
                    targets=("q", "v"), verbose=True)
    elif args.finetune == "top_k":
        for q in model.base.parameters():
            q.requires_grad = False
        for blk in model.base.transformer.h[-args.top_k:]:
            for q in blk.parameters():
                q.requires_grad = True
    elif args.finetune == "head_only":
        for q in model.base.parameters():
            q.requires_grad = False

    if args.finetune != "full":
        for q in model.head.parameters():
            q.requires_grad = True
        if getattr(model, "score", None) is not None:
            for q in model.score.parameters():
                q.requires_grad = True

    model = model.to(device)
    t, a = model.trainable_parameters()
    print(f"arch {arch} | finetune {args.finetune} | "
          f"trainable {t:,}/{a:,} ({100*t/a:.2f}%)")

    spe = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total = spe * args.epochs
    warmup = max(1, int(args.warmup_frac * total))
    print(f"{spe} steps/epoch x {args.epochs} = {total} steps\n")

    params = [q for q in model.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    run_dir = os.path.join(ROOT, "checkpoints/lsi", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_log.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    best, step, t0 = -1.0, 0, time.time()
    run_loss, nb = 0.0, 0
    lr = args.max_lr

    def do_eval(epoch):
        nonlocal best
        prob, gold = predict(model, va, args.batch_size, device, ctx)
        m05, micro, subset, _ = scores(prob, gold, 0.5)
        mg, tg = best_global(prob, gold)
        mpl, thr = best_per_label(prob, gold)
        flag = "  <- best" if mpl > best else ""
        print(f"  ep {epoch} step {step:5d} | loss {run_loss/max(1,nb):.4f} "
              f"| macro@.5 {m05:.4f} | global {mg:.4f}@{tg:.2f} "
              f"| per-label {mpl:.4f} | micro {micro:.4f}{flag}")
        with open(os.path.join(run_dir, "train_log.csv"), "a",
                  newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "epoch": epoch, "step": step,
                "train_loss": round(run_loss / max(1, nb), 5),
                "macro_05": round(m05, 5), "macro_global": round(mg, 5),
                "thr_global": round(tg, 3),
                "macro_perlabel": round(mpl, 5), "micro": round(micro, 5),
                "subset": round(subset, 5), "lr": f"{lr:.3e}",
                "elapsed_min": round((time.time() - t0) / 60, 2)})
        if mpl > best:
            best = mpl
            torch.save({"model": model.state_dict(), "arch": arch,
                        "step": step, "epoch": epoch, "macro_f1": mpl,
                        "thresholds": thr.tolist(), "thr_global": tg,
                        "macro_05": m05, "macro_global": mg,
                        "args": vars(args)},
                       os.path.join(run_dir, "ckpt_best.pt"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        for chunk, (ids, msk, y) in batches(tr, args.batch_size, device,
                                            seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(ids, msk, labels=y, pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_loss += loss.item()
            nb += 1
            step += 1
            if step % args.eval_every == 0:
                do_eval(epoch)
                run_loss, nb = 0.0, 0
        do_eval(epoch)

    print(f"\nbest dev macro-F1 (per-label thresholds) {best:.4f}")


if __name__ == "__main__":
    main()