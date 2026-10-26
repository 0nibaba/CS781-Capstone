"""
Fine-tune a Phase 1 model for L-NER.

Cross-validation across the two given folds: train on one, validate on the
other, then swap. check_folds.py verified the folds are comparable (label
ratios near 1.0 except A.COUNSEL and WIT at 1.79x) and free of cross-fold
near-duplicates (max Jaccard 0.005), so CV here is honest — unlike the Phase 1
eval set before contamination filtering.

Model selection is on validation macro-F1, NOT loss. Phase 1 established that
the two diverge, and macro-F1 is what is actually scored.
"""

import os
import csv
import json
import math
import time
import argparse

import torch
from tokenizers import Tokenizer

from bio_tagging import TAGS, bio_to_spans
from windowing import merge_windows
from ner_model import TokenClassifier, load_base, IGNORE
from ner_data import (NERWindows, load_fold, batches, collate,
                      class_weights, TOKENIZER)
from ner_metrics import strict_scores, report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FIELDS = ["epoch", "step", "train_loss", "val_loss", "macro_f1",
              "macro_p", "macro_r", "micro_f1", "lr", "elapsed_min"]


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


@torch.no_grad()
def predict(model, ds, batch_size, device, ctx, weights=None):
    """Run the model over a fold's windows and decode to character spans."""
    model.eval()
    per_doc = {}
    losses = []

    for chunk, (ids, msk, lab, pos) in batches(ds, batch_size, device,
                                               shuffle=False):
        with ctx:
            logits, loss = model(ids, attn_mask=msk, labels=lab,
                                 class_weights=weights, doc_pos=pos)
        if loss is not None:
            losses.append(loss.item())

        if model.use_crf:
            # the CRF's own decode: the best VALID sequence, not the argmax
            # at each position independently
            paths = model.crf.viterbi(logits.float(), msk)
        else:
            argmax = logits.argmax(-1).cpu()

        for i, it in enumerate(chunk):
            n = len(it["input_ids"])
            tags = paths[i][:n] if model.use_crf else argmax[i, :n].tolist()
            per_doc.setdefault(it["doc_id"], []).append({
                "tags": tags,
                "tok_start": it["tok_start"],
                "keep_from": it["keep_from"],
                "keep_to": it["keep_to"],
                "n_tokens": it["n_tokens"],
            })

    spans = {}
    for did, wins in per_doc.items():
        n_tok = wins[0]["n_tokens"]
        merged = merge_windows(wins, n_tok)
        spans[did] = bio_to_spans(merged, ds.offsets[did], ds.docs[did])

    model.train()
    return spans, (sum(losses) / len(losses) if losses else float("nan"))


def run_fold(args, train_fold, val_fold, tok, device, ctx):
    print(f"\n{'=' * 70}")
    print(f"train on {train_fold}, validate on {val_fold}")
    print(f"{'=' * 70}")

    tr_docs, tr_gold = load_fold(train_fold)
    va_docs, va_gold = load_fold(val_fold)
    tr = NERWindows(tr_docs, tr_gold, tok)
    va = NERWindows(va_docs, va_gold, tok)
    print(f"  {len(tr_docs)} train docs -> {len(tr)} windows")
    print(f"  {len(va_docs)} val docs   -> {len(va)} windows")

    gold_spans = {d: [[e[0], e[1], e[2]] for e in v]
                  for d, v in va_gold.items()}

    base, arch, d_model = load_base(args.checkpoint, device)
    model = TokenClassifier(base, arch, d_model,
                            dropout=args.dropout, use_position = args.use_position, use_crf=args.use_crf).to(device)

    if args.lorma:
        from lorma import apply_lorma
        apply_lorma(model, r=args.lorma_r, alpha=args.lorma_alpha)
        model.to(device)
    
    t, a = model.trainable_parameters()
    print(f"  arch {arch} | trainable {t:,} / {a:,} ({100*t/a:.2f}%)")

    weights = (class_weights(tr, len(TAGS), device, args.class_weights)
               if args.class_weights != "none" else None)
    
    if args.use_crf and weights is not None:
        raise ValueError("--class_weights has no effect with --use_crf: the "
                         "CRF loss is a sequence likelihood, not a sum of "
                         "per-token terms")

    steps_per_epoch = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total_steps))
    print(f"  {steps_per_epoch} steps/epoch x {args.epochs} epochs "
          f"= {total_steps} steps (warmup {warmup})")

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    run_dir = os.path.join(ROOT, "checkpoints/ner", f"{args.run_name}_{val_fold}")
    os.makedirs(run_dir, exist_ok=True)
    log_csv = os.path.join(run_dir, "train_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    best_f1, best_epoch = -1.0, -1
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0

        for chunk, (ids, msk, lab, pos) in batches(tr, args.batch_size, device,
                                              seed=args.seed + epoch):
            lr = get_lr(step, warmup, total_steps, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(ids, attn_mask=msk, labels=lab,
                                class_weights=weights, doc_pos = pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip)
            opt.step()

            ep_loss += loss.item()
            nb += 1
            step += 1

        pred, val_loss = predict(model, va, args.batch_size, device, ctx,
                                 weights)
        mf, mp, mr, per, micro = strict_scores(pred, gold_spans)

        print(f"  epoch {epoch:3d} | train {ep_loss/nb:.4f} | "
              f"val {val_loss:.4f} | macro-F1 {mf:.4f} "
              f"(P {mp:.3f} R {mr:.3f}) | micro {micro[2]:.4f}"
              f"{'  <- best' if mf > best_f1 else ''}")

        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "epoch": epoch, "step": step,
                "train_loss": round(ep_loss / nb, 5),
                "val_loss": round(val_loss, 5),
                "macro_f1": round(mf, 5), "macro_p": round(mp, 5),
                "macro_r": round(mr, 5), "micro_f1": round(micro[2], 5),
                "lr": f"{lr:.3e}",
                "elapsed_min": round((time.time() - t0) / 60, 2),
            })

        # selection on macro-F1, not loss: Phase 1 showed they diverge
        if mf > best_f1:
            best_f1, best_epoch = mf, epoch
            torch.save({"head": model.head.state_dict(),
                        "model": model.state_dict(),
                        "arch": arch, "epoch": epoch, "macro_f1": mf,
                        "args": vars(args)},
                       os.path.join(run_dir, "ckpt_best.pt"))
            with open(os.path.join(run_dir, "predictions.json"), "w") as f:
                json.dump(pred, f)

    print(f"\n  best macro-F1 {best_f1:.4f} at epoch {best_epoch}")
    with open(os.path.join(run_dir, "predictions.json")) as f:
        best_pred = json.load(f)
    report(best_pred, gold_spans, f"best epoch ({val_fold})")
    return best_f1, best_epoch

def train_all(args, tok, device, ctx):
    """Train on all 70 documents. No validation — the epoch count must come
    from a cross-validated run, since there is nothing here to select on."""
    print(f"\n{'=' * 70}")
    print(f"training on fold1 + fold2, {args.epochs} epochs (no validation)")
    print(f"{'=' * 70}")

    docs, gold = {}, {}
    for f in ("fold1", "fold2"):
        d, g = load_fold(f)
        docs.update(d)
        gold.update(g)
    tr = NERWindows(docs, gold, tok)
    print(f"  {len(docs)} documents -> {len(tr)} windows")

    base, arch, d_model = load_base(args.checkpoint, device)
    model = TokenClassifier(base, arch, d_model, dropout=args.dropout,
                            use_position=args.use_position,
                            use_crf=args.use_crf).to(device)

    if args.lorma:
        from lorma import apply_lorma
        apply_lorma(model, r=args.lorma_r, alpha=args.lorma_alpha)
        model.to(device)

    weights = (class_weights(tr, len(TAGS), device, args.class_weights)
               if args.class_weights != "none" else None)
    if args.use_crf and weights is not None:
        raise ValueError("--class_weights has no effect with --use_crf")

    steps_per_epoch = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total_steps))

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    run_dir = os.path.join(ROOT, "checkpoints/ner", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for chunk, (ids, msk, lab, pos) in batches(tr, args.batch_size, device,
                                                   seed=args.seed + epoch):
            lr = get_lr(step, warmup, total_steps, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(ids, attn_mask=msk, labels=lab,
                                class_weights=weights, doc_pos=pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip)
            opt.step()
            ep_loss += loss.item()
            nb += 1
            step += 1
        print(f"  epoch {epoch:3d} | train {ep_loss/nb:.4f}")

    torch.save({"model": model.state_dict(), "arch": arch,
                "epoch": args.epochs, "args": vars(args)},
               os.path.join(run_dir, "ckpt_final.pt"))
    print(f"\n  saved -> {run_dir}/ckpt_final.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="a Phase 1 checkpoint")
    p.add_argument("--tokenizer", default=TOKENIZER)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=3e-6)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--class_weights", default="none",
                   choices=["none", "inv_sqrt", "inv"])
    p.add_argument("--seed", type=int, default=1899)
    p.add_argument("--single_fold", default="",
                   help="run only this validation fold (fold1 or fold2)")
    p.add_argument("--use_position", action = "store_true", help = "give the head normalised document position")
    p.add_argument("--use_crf", action="store_true", help="CRF head: score whole tag sequences and decode with "
                        "Viterbi, instead of independent per-token argmax")
    p.add_argument("--all_data", action="store_true",
                   help="train on fold1+fold2 with no validation, for the "
                        "submission model. Epochs must be chosen from a prior "
                        "cross-validated run.")
    p.add_argument("--lorma", action="store_true")
    p.add_argument("--lorma_r", type=int, default=8)
    p.add_argument("--lorma_alpha", type=float, default=16.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    tok = Tokenizer.from_file(args.tokenizer)

    pairs = [("fold1", "fold2"), ("fold2", "fold1")]
    if args.single_fold:
        pairs = [(("fold2" if args.single_fold == "fold1" else "fold1"),
                  args.single_fold)]

    if args.all_data:
        train_all(args, tok, device, ctx)
        return
    
    results = []
    for tr_f, va_f in pairs:
        f1, ep = run_fold(args, tr_f, va_f, tok, device, ctx)
        results.append((va_f, f1, ep))

    print(f"\n{'=' * 70}")
    print("cross-validation summary")
    print(f"{'=' * 70}")
    for va_f, f1, ep in results:
        print(f"  val {va_f}: macro-F1 {f1:.4f} (epoch {ep})")
    if len(results) > 1:
        m = sum(r[1] for r in results) / len(results)
        spread = abs(results[0][1] - results[1][1])
        print(f"\n  mean {m:.4f}, spread {spread:.4f}")
        print("  report both folds, not just the mean — A.COUNSEL and WIT")
        print("  differ 1.79x between folds and will drive the variance")


if __name__ == "__main__":
    main()