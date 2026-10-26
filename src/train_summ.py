"""Fine-tune a Phase 1 model for abstractive legal summarization.

Trains on chunk 0 only by default: BERTScore truncates candidate and reference
at 512 RoBERTa tokens, so only the opening of each summary is ever scored, and
later chunks cannot affect the primary metric. --all_chunks trains on every
chunk pair instead, which serves coverage and the secondary ROUGE metrics; that
is a different experiment and is labelled as one.

Validation tracks loss every eval_every steps (cheap) and generates real
summaries on --gen_n dev documents at each epoch end (expensive but the only
thing that measures the metric we are judged on). Baseline to beat: lead-512
extraction scores BERTScore F1 0.8300.
"""
import os, csv, json, math, time, argparse

import numpy as np
import torch
from tokenizers import Tokenizer

from summ_data import SummPairs, load_split, batches, TOKENIZER, MAX_SRC
from summ_model import Summarizer, strip_eos
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FIELDS = ["epoch", "step", "train_loss", "val_loss", "bert_f1",
              "gen_tokens", "lr", "elapsed_min"]


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


@torch.no_grad()
def val_loss(model, ds, bs, device, ctx, arch, chunk0, limit=200):
    model.eval()
    tot, n = 0.0, 0
    for c, b in batches(ds, bs, device, arch, shuffle=False,
                        chunk0_only=chunk0):
        with ctx:
            _, loss = model(b)
        tot += loss.item(); n += 1
        if n >= limit:
            break
    model.train()
    return tot / max(1, n)


@torch.no_grad()
def gen_score(model, docs, tok, bs, device, ctx, arch, scorer,
              max_len, min_len):
    model.eval()
    ds = SummPairs(docs, tok, with_target=False)
    first = [it for it in ds.items if it["chunk"] == 0]
    cands, lens = [], []
    for i in range(0, len(first), bs):
        c = first[i:i + bs]
        from summ_data import collate
        b = collate(c, device, arch)
        with ctx:
            out = model.generate(b, max_len=max_len, min_len=min_len)
        txt = strip_eos(out, tok)
        cands.extend(txt)
        lens.extend(len(tok.encode(t).ids) for t in txt)
    refs = [d["summary"] for d in docs]
    P, R, F = scorer.score(cands, refs)
    model.train()
    return F.mean().item(), float(np.mean(lens)), cands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--finetune", default="full",
                   choices=["full", "lorma", "top_k", "head_only"])
    p.add_argument("--lorma_r", type=int, default=8)
    p.add_argument("--lorma_alpha", type=float, default=16.0)
    p.add_argument("--top_k_layers", type=int, default=4)
    p.add_argument("--all_chunks", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_batch", type=int, default=8)
    p.add_argument("--gen_n", type=int, default=50)
    p.add_argument("--max_len", type=int, default=544)
    p.add_argument("--min_len", type=int, default=480)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--max_lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=3e-6)
    p.add_argument("--warmup_frac", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1899)
    p.add_argument("--limit_train", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    tok = Tokenizer.from_file(args.tokenizer)
    tr_docs = load_split("train")
    if args.limit_train:
        tr_docs = tr_docs[:args.limit_train]
    dv_docs = load_split("dev")

    t0 = time.time()
    tr = SummPairs(tr_docs, tok)
    va = SummPairs(dv_docs[:200], tok)
    chunk0 = not args.all_chunks
    n_tr = sum(1 for it in tr.items if it["chunk"] == 0 or args.all_chunks)
    print(f"tokenised in {time.time()-t0:.0f}s | train docs {len(tr_docs)} "
          f"-> {n_tr} pairs ({'chunk0' if chunk0 else 'all chunks'})")

    base, arch, d_model = load_base(args.checkpoint, device)
    model = Summarizer(base, arch)

    if args.finetune == "lorma":
        from lorma import apply_lorma
        apply_lorma(model.base, r=args.lorma_r, alpha=args.lorma_alpha,
                    targets=("q", "v"), verbose=True)
    elif args.finetune == "top_k":
        for q in model.base.parameters():
            q.requires_grad = False
        stack = (model.base.decoder if arch == "encdec"
                 else model.base.transformer.h)
        for blk in stack[-args.top_k_layers:]:
            for q in blk.parameters():
                q.requires_grad = True
    elif args.finetune == "head_only":
        for q in model.base.parameters():
            q.requires_grad = False
        model.base.lm_head.weight.requires_grad = True

    model = model.to(device)
    t, a = model.trainable_parameters()
    print(f"arch {arch} | finetune {args.finetune} | "
          f"trainable {t:,}/{a:,} ({100*t/a:.2f}%)")

    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=False)

    spe = max(1, -(-n_tr // args.batch_size))
    total = spe * args.epochs
    warmup = max(1, int(args.warmup_frac * total))
    print(f"{spe} steps/epoch x {args.epochs} = {total} steps")
    print("baseline to beat: lead-512 extraction = 0.8300 BERTScore F1\n")

    params = [q for q in model.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    run_dir = os.path.join(ROOT, "checkpoints/summ", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_log.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    best, step, t0 = -1.0, 0, time.time()
    run_loss, nb = 0.0, 0
    lr = args.max_lr

    for epoch in range(1, args.epochs + 1):
        model.train()
        for c, b in batches(tr, args.batch_size, device, arch,
                            seed=args.seed + epoch, chunk0_only=chunk0):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_loss += loss.item(); nb += 1; step += 1

            if step % args.eval_every == 0:
                vl = val_loss(model, va, args.batch_size, device, ctx, arch,
                              chunk0)
                print(f"  ep {epoch} step {step:5d} | train "
                      f"{run_loss/nb:.4f} | val {vl:.4f}")
                run_loss, nb = 0.0, 0

        bf, gl, cands = gen_score(model, dv_docs[:args.gen_n], tok,
                                  args.gen_batch, device, ctx, arch, scorer,
                                  args.max_len, args.min_len)
        vl = val_loss(model, va, args.batch_size, device, ctx, arch, chunk0)
        flag = "  <- best" if bf > best else ""
        print(f"  ep {epoch} END    | val {vl:.4f} | BERTScore {bf:.4f} "
              f"| gen len {gl:.0f}{flag}")
        with open(os.path.join(run_dir, "train_log.csv"), "a",
                  newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "epoch": epoch, "step": step,
                "train_loss": round(run_loss / max(1, nb), 5),
                "val_loss": round(vl, 5), "bert_f1": round(bf, 5),
                "gen_tokens": round(gl, 1), "lr": f"{lr:.3e}",
                "elapsed_min": round((time.time() - t0) / 60, 2)})
        if bf > best:
            best = bf
            torch.save({"model": model.state_dict(), "arch": arch,
                        "epoch": epoch, "bert_f1": bf, "args": vars(args)},
                       os.path.join(run_dir, "ckpt_best.pt"))
            with open(os.path.join(run_dir, "sample.txt"), "w") as f:
                for i, c_ in enumerate(cands[:3]):
                    f.write(f"--- {i} ---\n{c_}\n\n")

    print(f"\nbest dev BERTScore F1 {best:.4f}  (lead-512 baseline 0.8300)")


if __name__ == "__main__":
    main()