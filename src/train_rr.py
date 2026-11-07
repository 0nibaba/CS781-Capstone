import os
import csv
import json
import math
import time
import argparse
from collections import Counter

import torch
from tokenizers import Tokenizer

from rr_data import (RRBlocks, load_split, batches, TOKENIZER,
                     LABELS, ID2LABEL, N_LABELS, IGNORE)
from rr_metrics import macro_scores, report
from rr_model import SentenceClassifier
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FIELDS = ["epoch", "step", "train_loss", "macro_f1", "macro_p", "macro_r",
              "accuracy", "cl_f1", "it_f1", "lr", "elapsed_min"]


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


def class_weights(ds, device, scheme="inv_sqrt"):
    c = torch.zeros(N_LABELS)
    for it in ds.items:
        for si in range(it["pred_from"], it["pred_to"]):
            if it["labels"][si] != IGNORE:
                c[it["labels"][si]] += 1
    c = c.clamp(min=1.0)
    if scheme == "none":
        w = torch.ones(N_LABELS)
    elif scheme == "inv":
        w = c.sum() / c
    else:
        w = (c.sum() / c).sqrt()
    return (w / w.mean()).to(device)


@torch.no_grad()
def predict(model, ds, batch_size, device, ctx):
    model.eval()
    out = {did: [None] * len(d["text"]) for did, d in ds.docs.items()}

    for chunk, (ids, msk, pos, pmask, lab) in batches(ds, batch_size, device,
                                                      shuffle=False):
        with ctx:
            logits, _ = model(ids, msk, pos, pmask)
        paths = model.decode(logits, pmask)
        for b, it in enumerate(chunk):
            span = range(it["pred_from"], it["pred_to"])
            for j, si in enumerate(span):
                out[it["doc_id"]][si] = ID2LABEL[paths[b][j]]

    missing = [(d, i) for d, v in out.items()
               for i, x in enumerate(v) if x is None]
    assert not missing, f"{len(missing)} sentences never predicted"
    model.train()
    return out


def evaluate(model, ds, docs, batch_size, device, ctx):
    pred = predict(model, ds, batch_size, device, ctx)
    gold = {d["id"]: d["labels"] for d in docs}
    mf, mp, mr, acc, per = macro_scores(pred, gold)

    by_dom = {}
    for dom in ("CL", "IT"):
        g = {d["id"]: d["labels"] for d in docs if d["domain"] == dom}
        if g:
            p = {k: pred[k] for k in g}
            by_dom[dom] = macro_scores(p, g)[0]
    return mf, mp, mr, acc, per, by_dom, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint", required=True, help="a Phase 1 checkpoint")
    p.add_argument("--tokenizer", default=TOKENIZER)

    p.add_argument("--use_crf", action="store_true",
                   help="CRF over the sentence label sequence; the 94%/75% "
                        "self-transition rate is what this exploits")
    p.add_argument("--class_weights", default="none",
                   choices=["none", "inv_sqrt", "inv"])
    p.add_argument("--domains", default="CL,IT")
    p.add_argument("--all_data", action="store_true",
                   help="train on train+dev for submission; no validation, so "
                        "--epochs must come from a validated run")

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=3e-6)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--context", type=int, default=8,
                   help="lead-in sentences encoded but not predicted")
    p.add_argument("--max_len", type=int, default=1024,
                   help="block budget; cannot exceed the pretrained block size")
    p.add_argument("--seed", type=int, default=1899)
    p.add_argument("--use_lorma", action="store_true")
    p.add_argument("--lorma_r", type=int, default=8)
    p.add_argument("--lorma_alpha", type=float, default=16.0)
    p.add_argument("--ctx_frac", type=float, default=0.34,
                   help="max share of the block budget given to lead-in")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    doms = tuple(args.domains.split(","))
    tok = Tokenizer.from_file(args.tokenizer)

    train_docs = load_split("train", doms)
    dev_docs = load_split("dev", doms)
    if args.all_data:
        train_docs = train_docs + dev_docs

    tr = RRBlocks(train_docs, tok, max_len=args.max_len,
                  context=args.context, ctx_frac=args.ctx_frac)
    va = RRBlocks(dev_docs, tok, max_len=args.max_len,
                  context=args.context, ctx_frac=args.ctx_frac)
    print(f"train {len(train_docs)} docs -> {len(tr)} blocks")
    print(f"dev   {len(dev_docs)} docs -> {len(va)} blocks"
          + ("   (also in training — --all_data)" if args.all_data else ""))

    base, arch, d_model = load_base(args.checkpoint, device)
    model = SentenceClassifier(base, arch, d_model, dropout=args.dropout,
                               use_crf=args.use_crf)

    if args.use_lorma:
        from lorma import apply_lorma
        apply_lorma(model.base, r=args.lorma_r, alpha=args.lorma_alpha,
                    targets=("q", "v"), verbose=True)
        for q in model.head.parameters():
            q.requires_grad = True
        if getattr(model, "crf", None) is not None:
            for q in model.crf.parameters():
                q.requires_grad = True

    model = model.to(device)
    t, a = model.trainable_parameters()
    print(f"arch {arch} | crf {args.use_crf} | lorma {args.use_lorma} | "
          f"trainable {t:,} / {a:,} ({100*t/a:.2f}%)")

    weights = None
    if args.class_weights != "none":
        if args.use_crf:
            raise ValueError("--class_weights has no effect with --use_crf: "
                             "the CRF loss is a sequence likelihood")
        weights = class_weights(tr, device, args.class_weights)

    spe = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total = spe * args.epochs
    warmup = max(1, int(args.warmup_frac * total))
    print(f"{spe} steps/epoch x {args.epochs} epochs = {total} steps\n")

    params = [q for q in model.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    run_dir = os.path.join(ROOT, "checkpoints/rr", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_log.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    best, best_ep, step = -1.0, -1, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for chunk, (ids, msk, pos, pmask, lab) in batches(
                tr, args.batch_size, device, seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(ids, msk, pos, pmask, labels=lab,
                                class_weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            ep_loss += loss.item()
            nb += 1
            step += 1

        mf, mp, mr, acc, per, by_dom, pred = evaluate(
            model, va, dev_docs, args.batch_size, device, ctx)
        flag = "  <- best" if mf > best else ""
        print(f"  epoch {epoch:3d} | train {ep_loss/nb:9.4f} | "
              f"macro-F1 {mf:.4f} | acc {acc:.4f} | "
              f"CL {by_dom.get('CL', float('nan')):.4f} "
              f"IT {by_dom.get('IT', float('nan')):.4f}{flag}")

        with open(os.path.join(run_dir, "train_log.csv"), "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "epoch": epoch, "step": step,
                "train_loss": round(ep_loss / nb, 5),
                "macro_f1": round(mf, 5), "macro_p": round(mp, 5),
                "macro_r": round(mr, 5), "accuracy": round(acc, 5),
                "cl_f1": round(by_dom.get("CL", 0), 5),
                "it_f1": round(by_dom.get("IT", 0), 5),
                "lr": f"{lr:.3e}",
                "elapsed_min": round((time.time() - t0) / 60, 2),
            })

        if mf > best or args.all_data:
            best, best_ep = max(mf, best), epoch
            torch.save({"model": model.state_dict(), "arch": arch,
                        "epoch": epoch, "macro_f1": mf, "args": vars(args)},
                       os.path.join(run_dir,
                                    "ckpt_final.pt" if args.all_data
                                    else "ckpt_best.pt"))
            if not args.all_data:
                with open(os.path.join(run_dir, "predictions.json"), "w") as f:
                    json.dump(pred, f)

    print(f"\nbest dev macro-F1 {best:.4f} at epoch {best_ep}")
    print("reference: majority 0.029, copy-previous-gold oracle 0.754")

    if not args.all_data:
        with open(os.path.join(run_dir, "predictions.json")) as f:
            bp = json.load(f)
        report(bp, {d["id"]: d["labels"] for d in dev_docs}, "best epoch")


if __name__ == "__main__":
    main()