import os, csv, json, math, time, argparse

import numpy as np
import torch
from tokenizers import Tokenizer

from pcr_baseline import (load_split, micro_f1_at_k, ceiling, bm25_rank,
                          TOKENIZER)
from pcr_data import Corpus, PairSampler, pack
from pcr_model import CaseEncoder
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FIELDS = ["epoch", "step", "loss", "f1", "p", "r", "lr", "elapsed_min"]


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)


@torch.no_grad()
def embed_all(model, chunks, device, ctx, batch=32):
    out = []
    order = np.argsort([len(c) for c in chunks])
    E = torch.zeros(len(chunks), model.embed_dim, dtype=torch.float32)
    for i in range(0, len(order), batch):
        sel = order[i:i + batch]
        ids, msk = pack([chunks[j] for j in sel], device)
        with ctx:
            v = model.embed(ids, msk)
        E[sel] = v.cpu()
    return E


@torch.no_grad()
def rank_dev(model, qcorp, ccorp, queries, device, ctx, batch=32):
    model.eval()
    cflat = ccorp.flat()
    cpos = {k: i for i, k in enumerate(ccorp.ids)}
    cowner = torch.tensor([cpos[k] for k, _ in cflat])
    CE = embed_all(model, [c for _, c in cflat], device, ctx, batch)

    ranked = {}
    for q in qcorp.ids:
        QE = embed_all(model, qcorp.chunks[q], device, ctx, batch)
        S = QE @ CE.T                              # (nq_chunks, nc_chunks)
        best_chunk = S.max(0).values                # best query chunk
        per_doc = torch.full((len(ccorp.ids),), -1e9)
        per_doc = per_doc.scatter_reduce(
            0, cowner, best_chunk, reduce="amax", include_self=True)
        ranked[q] = [ccorp.ids[i]
                     for i in torch.argsort(-per_doc).tolist()]
    model.train()
    return micro_f1_at_k(ranked, queries), ranked


def mine_hard(queries, cands, tok, n=20, budget=1024):
    r = bm25_rank(queries, cands, tok, max_tokens=budget)
    out = {}
    for q, v in queries.items():
        gold = set(v["gold"])
        out[q] = [c for c in r[q] if c not in gold][:n]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--finetune", default="full",
                   choices=["full", "lorma", "top_k", "ia3", "head_only"])
    p.add_argument("--lorma_r", type=int, default=8)
    p.add_argument("--lorma_alpha", type=float, default=16.0)
    p.add_argument("--top_k_layers", type=int, default=4)
    p.add_argument("--pooling", default="mean",
                   choices=["mean", "last", "attn"])
    p.add_argument("--hard_negatives", action="store_true")
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_lr", type=float, default=2e-5)
    p.add_argument("--min_lr", type=float, default=2e-6)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1899)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    tok = Tokenizer.from_file(args.tokenizer)
    t0 = time.time()
    trq, trc = load_split("train")
    dvq, dvc = load_split("dev")
    tr_qcorp = Corpus({k: v["text"] for k, v in trq.items()}, tok)
    tr_ccorp = Corpus(trc, tok)
    dv_qcorp = Corpus({k: v["text"] for k, v in dvq.items()}, tok)
    dv_ccorp = Corpus(dvc, tok)
    print(f"tokenised in {time.time()-t0:.0f}s | train {len(trq)} queries / "
          f"{len(trc)} candidates | dev {len(dvq)} / {len(dvc)}")

    hard = None
    if args.hard_negatives:
        t0 = time.time()
        hard = mine_hard(trq, trc, tok)
        print(f"mined BM25 hard negatives in {time.time()-t0:.0f}s "
              f"({np.mean([len(v) for v in hard.values()]):.1f}/query)")

    sampler = PairSampler(trq, tr_qcorp, tr_ccorp, seed=args.seed)
    print(f"{len(sampler)} (query, positive) pairs")

    base, arch, d_model = load_base(args.checkpoint, device)
    model = CaseEncoder(base, arch, d_model, pooling=args.pooling)
    model.embed_dim = d_model

    if args.finetune == "lorma":
        from lorma import apply_lorma
        apply_lorma(model.base, r=args.lorma_r, alpha=args.lorma_alpha,
                    targets=("q", "v"), verbose=True)
    elif args.finetune == "top_k":
        for q in model.base.parameters():
            q.requires_grad = False
        stack = (model.base.encoder if arch == "encdec"
                 else model.base.transformer.h)
        for blk in stack[-args.top_k_layers:]:
            for q in blk.parameters():
                q.requires_grad = True
    elif args.finetune == "ia3":
        for q in model.base.parameters():
            q.requires_grad = False
        from ia3 import apply_ia3
        apply_ia3(model.base, verbose=True)
    elif args.finetune == "head_only":
        for q in model.base.parameters():
            q.requires_grad = False

    model = model.to(device)
    t, a = model.trainable_parameters()
    print(f"arch {arch} | finetune {args.finetune} | pooling {args.pooling} "
          f"| hard_neg {args.hard_negatives} | trainable {t:,}/{a:,} "
          f"({100*t/a:.2f}%)")

    c = ceiling(dvq)
    print(f"\ndev ceiling F1 {c[0]:.4f} | BM25@4096 0.2991 | "
          f"zero-shot dense 0.1946\n")

    params = [q for q in model.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    spe = max(1, len(sampler) // args.batch_size)
    total = spe * args.epochs
    warmup = max(1, int(args.warmup_frac * total))

    run_dir = os.path.join(ROOT, "checkpoints/pcr", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_log.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    (f1, pr, rc), _ = rank_dev(model, dv_qcorp, dv_ccorp, dvq, device, ctx)
    print(f"  epoch 0 (untrained) | F1@7 {f1:.4f}")

    best, step, t0 = f1, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for keep, (qi, qm), (ci, cm), neg in sampler.batches(
                args.batch_size, device, epoch=epoch, hard=hard):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                qv = model.embed(qi, qm)
                cv = model.embed(ci, cm)
                nv = model.embed(*neg) if neg is not None else None
                loss = model.info_nce(qv, cv, temp=args.temp, neg=nv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run += loss.item(); nb += 1; step += 1

        (f1, pr, rc), ranked = rank_dev(model, dv_qcorp, dv_ccorp, dvq,
                                        device, ctx)
        flag = "  <- best" if f1 > best else ""
        print(f"  epoch {epoch} | loss {run/max(1,nb):.4f} | F1@7 {f1:.4f} "
              f"| P {pr:.4f} R {rc:.4f}{flag}")
        with open(os.path.join(run_dir, "train_log.csv"), "a",
                  newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "epoch": epoch, "step": step,
                "loss": round(run / max(1, nb), 5), "f1": round(f1, 5),
                "p": round(pr, 5), "r": round(rc, 5), "lr": f"{lr:.3e}",
                "elapsed_min": round((time.time() - t0) / 60, 2)})
        if f1 > best:
            best = f1
            torch.save({"model": model.state_dict(), "arch": arch,
                        "epoch": epoch, "f1": f1, "args": vars(args)},
                       os.path.join(run_dir, "ckpt_best.pt"))

    print(f"\nbest dev F1@7 {best:.4f}  "
          f"(BM25@4096 0.2991, ceiling {c[0]:.4f})")


if __name__ == "__main__":
    main()