import os, csv, json, math, time, argparse
from collections import Counter

import torch
from tokenizers import Tokenizer

from cjpe_data import CJPEDocs, load_split, batches, TOKENIZER, MODES
from cjpe_model import DocClassifier, POOLINGS
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FIELDS = ["epoch", "step", "train_loss", "macro_f1", "macro_f1_adj",
              "acc", "f1_0", "f1_1", "pred_1_frac", "lr", "elapsed_min"]


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


def macro_f1(pred, gold):
    f1s = []
    for c in (0, 1):
        tp = sum(1 for p, g in zip(pred, gold) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred, gold) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred, gold) if p != c and g == c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    acc = sum(1 for p, g in zip(pred, gold) if p == g) / max(1, len(gold))
    return sum(f1s) / 2, acc, f1s


@torch.no_grad()
def evaluate(model, ds, batch_size, device, ctx, log_prior):
    model.eval()
    all_logits, gold = [], []
    for chunk, (ids, msk, lab) in batches(ds, batch_size, device,
                                          shuffle=False):
        with ctx:
            logits, _ = model(ids, msk)
        all_logits.append(logits.float().cpu())
        gold.extend(lab.cpu().tolist())
    L = torch.cat(all_logits)
    model.train()

    raw = L.argmax(-1).tolist()
    adj = (L - log_prior).argmax(-1).tolist()
    mf, acc, f1s = macro_f1(raw, gold)
    mf_adj, acc_adj, _ = macro_f1(adj, gold)
    return mf, mf_adj, acc, f1s, sum(raw) / len(raw), L, gold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--mode", default="tail", choices=MODES)
    p.add_argument("--pooling", default="last", choices=POOLINGS)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--class_weights", default="none",
                   choices=["none", "balanced"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=1500)
    p.add_argument("--max_lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=3e-6)
    p.add_argument("--warmup_frac", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1899)
    p.add_argument("--limit_train", type=int, default=0,
                   help="debug: cap training cases")
    p.add_argument("--finetune", default="full",
                   choices=["full", "lorma", "bitfit", "head_only"])
    p.add_argument("--lorma_r", type=int, default=8)
    p.add_argument("--lorma_alpha", type=float, default=16.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    tok = Tokenizer.from_file(args.tokenizer)
    train_docs = load_split("train")
    dev_docs = load_split("dev")
    if args.limit_train:
        train_docs = train_docs[:args.limit_train]

    t0 = time.time()
    tr = CJPEDocs(train_docs, tok, mode=args.mode, max_len=args.max_len)
    va = CJPEDocs(dev_docs, tok, mode=args.mode, max_len=args.max_len)
    print(f"tokenised in {time.time()-t0:.0f}s | "
          f"train {len(tr)} | dev {len(va)} | mode {args.mode}")

    c = Counter(it["label"] for it in tr.items)
    n = sum(c.values())
    prior = torch.tensor([c[0] / n, c[1] / n])
    print(f"train prior {prior.tolist()} | dev prior "
          f"{[round(sum(1 for i in va.items if i['label']==k)/len(va), 3) for k in (0,1)]}")
    log_prior = prior.log()

    weights = None
    if args.class_weights == "balanced":
        w = (1.0 / prior)
        weights = (w / w.mean()).to(device)
        print(f"class weights {weights.tolist()}")

    base, arch, d_model = load_base(args.checkpoint, device)
    model = DocClassifier(base, arch, d_model, dropout=args.dropout,
                          pooling=args.pooling)

    if args.finetune == "lorma":
        from lorma import apply_lorma
        apply_lorma(model.base, r=args.lorma_r, alpha=args.lorma_alpha,
                    targets=("q", "v"), verbose=True)
    elif args.finetune == "bitfit":
        for n_, q in model.base.named_parameters():
            q.requires_grad = n_.endswith(".bias") or n_ == "bias"
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
    print(f"arch {arch} | pooling {args.pooling} | trainable {t:,}/{a:,} "
          f"({100*t/a:.2f}%)")

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

    run_dir = os.path.join(ROOT, "checkpoints/cjpe", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_log.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    best, best_tag, step, t0 = -1.0, "", 0, time.time()
    run_loss, nb = 0.0, 0

    def do_eval(epoch):
        nonlocal best, best_tag
        mf, mf_adj, acc, f1s, p1, _, _ = evaluate(
            model, va, args.batch_size, device, ctx, log_prior)
        score = max(mf, mf_adj)
        tag = "adj" if mf_adj > mf else "raw"
        flag = "  <- best" if score > best else ""
        print(f"  ep {epoch} step {step:5d} | loss {run_loss/max(1,nb):.4f} "
              f"| macro-F1 {mf:.4f} | +prior {mf_adj:.4f} | acc {acc:.4f} "
              f"| pred1 {p1:.3f}{flag}")
        with open(os.path.join(run_dir, "train_log.csv"), "a",
                  newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "epoch": epoch, "step": step,
                "train_loss": round(run_loss / max(1, nb), 5),
                "macro_f1": round(mf, 5), "macro_f1_adj": round(mf_adj, 5),
                "acc": round(acc, 5), "f1_0": round(f1s[0], 5),
                "f1_1": round(f1s[1], 5), "pred_1_frac": round(p1, 5),
                "lr": f"{lr:.3e}",
                "elapsed_min": round((time.time() - t0) / 60, 2)})
        if score > best:
            best, best_tag = score, tag
            torch.save({"model": model.state_dict(), "arch": arch,
                        "step": step, "epoch": epoch, "macro_f1": score,
                        "decision": tag, "args": vars(args)},
                       os.path.join(run_dir, "ckpt_best.pt"))

    lr = args.max_lr
    for epoch in range(1, args.epochs + 1):
        model.train()
        for chunk, (ids, msk, lab) in batches(tr, args.batch_size, device,
                                              seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(ids, msk, labels=lab, class_weights=weights)
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

    print(f"\nbest dev macro-F1 {best:.4f} (decision rule: {best_tag})")
    print("reference: majority-class macro-F1 0.3333 on a balanced dev set")


if __name__ == "__main__":
    main()