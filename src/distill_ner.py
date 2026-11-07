import os
import csv
import copy
import json
import math
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from bio_tagging import TAGS, bio_to_spans
from windowing import merge_windows
from ner_model import TokenClassifier, load_base, IGNORE
from ner_data import NERWindows, load_fold, batches, TOKENIZER
from ner_metrics import strict_scores, report
from train_ner import get_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_student(teacher, keep_every=2):
    student = copy.deepcopy(teacher)
    blocks = student.base.transformer.h
    kept = [blocks[i] for i in range(0, len(blocks), keep_every)]
    student.base.transformer.h = nn.ModuleList(kept)
    student.base.config.n_layer = len(kept)
    return student


@torch.no_grad()
def evaluate(model, ds, batch_size, device, ctx, gold):
    model.eval()
    per_doc = {}
    for chunk, (ids, msk, lab, pos) in batches(ds, batch_size, device,
                                               shuffle=False):
        with ctx:
            logits, _ = model(ids, attn_mask=msk, doc_pos=pos)
        paths = (model.crf.viterbi(logits.float(), msk) if model.use_crf
                 else None)
        argmax = None if model.use_crf else logits.argmax(-1).cpu()
        for i, it in enumerate(chunk):
            n = len(it["input_ids"])
            tags = paths[i][:n] if model.use_crf else argmax[i, :n].tolist()
            per_doc.setdefault(it["doc_id"], []).append({
                "tags": tags, "tok_start": it["tok_start"],
                "keep_from": it["keep_from"], "keep_to": it["keep_to"],
                "n_tokens": it["n_tokens"]})

    spans = {}
    for did, wins in per_doc.items():
        merged = merge_windows(wins, wins[0]["n_tokens"])
        spans[did] = bio_to_spans(merged, ds.offsets[did], ds.docs[did])
    model.train()
    return strict_scores(spans, gold), spans


def train_student(student, teacher, tr, va, gold_va, args, device, ctx,
                  distil, tag):
    print(f"\n{'-' * 66}")
    print(f"{tag}: {'WITH' if distil else 'WITHOUT'} distillation")
    print(f"{'-' * 66}")

    params = [p for p in student.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    print(f"  student {n:,} params")

    decay = [p for p in params if p.dim() >= 2]
    nodecay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    spe = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total = spe * args.epochs
    warmup = max(1, int(0.1 * total))

    best, best_ep, step = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        ep_h, ep_k, nb = 0.0, 0.0, 0

        for chunk, (ids, msk, lab, pos) in batches(tr, args.batch_size, device,
                                                   seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)

            with ctx:
                s_logits, hard = student(ids, attn_mask=msk, labels=lab,
                                         doc_pos=pos)
                loss = hard
                kd = torch.tensor(0.0, device=device)

                if distil:
                    with torch.no_grad():
                        t_logits, _ = teacher(ids, attn_mask=msk, doc_pos=pos)
                    T = args.temperature
                    valid = (lab != IGNORE).view(-1)
                    s = F.log_softmax(s_logits.float().view(-1, len(TAGS))[valid] / T, -1)
                    t = F.softmax(t_logits.float().view(-1, len(TAGS))[valid] / T, -1)
                    # T^2 keeps the gradient magnitude comparable to the hard loss
                    kd = F.kl_div(s, t, reduction="batchmean") * (T * T)
                    loss = (1 - args.alpha) * hard + args.alpha * kd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            ep_h += hard.item()
            ep_k += float(kd)
            nb += 1
            step += 1

        (mf, mp, mr, per, micro), _ = evaluate(student, va, args.batch_size,
                                               device, ctx, gold_va)
        flag = "  <- best" if mf > best else ""
        print(f"  epoch {epoch:3d} | hard {ep_h/nb:8.2f} | kd {ep_k/nb:7.4f} "
              f"| macro-F1 {mf:.4f} | micro {micro[2]:.4f}{flag}")
        if mf > best:
            best, best_ep = mf, epoch
            torch.save({"model": student.state_dict(), "epoch": epoch,
                        "macro_f1": mf, "args": vars(args),
                        "distilled": distil},
                       os.path.join(args.out_dir, f"{tag}_best.pt"))

    print(f"  best {best:.4f} at epoch {best_ep}")
    return best, best_ep, n

def distil_all(args, tok, device, ctx):
    docs, gold = {}, {}
    for f in ("fold1", "fold2"):
        d, g = load_fold(f)
        docs.update(d)
        gold.update(g)
    tr = NERWindows(docs, gold, tok)
    print(f"training on {len(docs)} documents -> {len(tr)} windows")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    teacher = TokenClassifier(base, arch, d_model, use_crf=True).to(device)
    ck_path = os.path.join(args.teacher_dir, "ckpt_final.pt")
    if not os.path.exists(ck_path):
        ck_path = os.path.join(args.teacher_dir, "ckpt_best.pt")
    teacher.load_state_dict(torch.load(ck_path, map_location="cpu",
                                       weights_only=False)["model"])
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad = False

    student = make_student(teacher, args.keep_every).to(device)
    for q in student.parameters():
        q.requires_grad = True
    n_s = sum(q.numel() for q in student.parameters())
    n_t = sum(q.numel() for q in teacher.parameters())
    print(f"teacher {n_t:,} -> student {n_s:,} "
          f"({100*n_s/n_t:.0f}%), {len(student.base.transformer.h)} layers")

    params = [q for q in student.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    spe = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total = spe * args.epochs
    warmup = max(1, int(0.1 * total))
    step = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        ep_h, ep_k, nb = 0.0, 0.0, 0
        for chunk, (ids, msk, lab, pos) in batches(tr, args.batch_size, device,
                                                   seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                s_logits, hard = student(ids, attn_mask=msk, labels=lab,
                                         doc_pos=pos)
                with torch.no_grad():
                    t_logits, _ = teacher(ids, attn_mask=msk, doc_pos=pos)
                T = args.temperature
                valid = (lab != IGNORE).view(-1)
                s = F.log_softmax(
                    s_logits.float().view(-1, len(TAGS))[valid] / T, -1)
                t = F.softmax(
                    t_logits.float().view(-1, len(TAGS))[valid] / T, -1)
                kd = F.kl_div(s, t, reduction="batchmean") * (T * T)
                loss = (1 - args.alpha) * hard + args.alpha * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            ep_h += hard.item()
            ep_k += kd.item()
            nb += 1
            step += 1
        print(f"  epoch {epoch:3d} | hard {ep_h/nb:8.2f} | kd {ep_k/nb:7.4f}")

    torch.save({"model": student.state_dict(), "arch": arch,
                "epoch": args.epochs, "args": vars(args)},
               os.path.join(args.out_dir, "ckpt_final.pt"))
    print(f"\nsaved -> {args.out_dir}/ckpt_final.pt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True,
                   help="checkpoints/ner/<run> holding the fine-tuned teacher")
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--val_fold", default="fold2", choices=["fold1", "fold2"])
    p.add_argument("--keep_every", type=int, default=2,
                   help="2 keeps 6 of 12 layers")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7,
                   help="weight on the distillation term")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1899)
    p.add_argument("--run_name", default="distil")
    p.add_argument("--all_data", action="store_true",
                   help="train the student on all 70 documents against an "
                        "all-data teacher; no validation, so --epochs must "
                        "come from a cross-validated run")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    args.out_dir = os.path.join(ROOT, "checkpoints/ner", args.run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = Tokenizer.from_file(TOKENIZER)

    if args.all_data:
        distil_all(args, tok, device, ctx)
        return

    train_fold = "fold1" if args.val_fold == "fold2" else "fold2"
    tr_docs, tr_gold = load_fold(train_fold)
    va_docs, va_gold = load_fold(args.val_fold)
    tr = NERWindows(tr_docs, tr_gold, tok)
    va = NERWindows(va_docs, va_gold, tok)
    gold_va = {d: [[e[0], e[1], e[2]] for e in v] for d, v in va_gold.items()}
    print(f"train {train_fold} ({len(tr)} windows), "
          f"val {args.val_fold} ({len(va)} windows)")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    teacher = TokenClassifier(base, arch, d_model, use_crf=True).to(device)
    ck_path = os.path.join(args.teacher_dir, "ckpt_final.pt")
    if not os.path.exists(ck_path):
        ck_path = os.path.join(args.teacher_dir, "ckpt_best.pt")
    teacher.load_state_dict(torch.load(ck_path, map_location="cpu",
                                       weights_only=False)["model"])
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad = False

    n_teacher = sum(q.numel() for q in teacher.parameters())
    (tf, _, _, _, tmicro), _ = evaluate(teacher, va, args.batch_size,
                                        device, ctx, gold_va)
    print(f"\nteacher: {n_teacher:,} params, "
          f"{len(teacher.base.transformer.h)} layers, "
          f"macro-F1 {tf:.4f} on {args.val_fold}")

    s1 = make_student(teacher, args.keep_every).to(device)
    s2 = copy.deepcopy(s1).to(device)
    for q in list(s1.parameters()) + list(s2.parameters()):
        q.requires_grad = True
    print(f"student: {sum(q.numel() for q in s1.parameters()):,} params, "
          f"{len(s1.base.transformer.h)} layers "
          f"({100*sum(q.numel() for q in s1.parameters())/n_teacher:.1f}% "
          f"of the teacher)")

    f_kd, e_kd, n_s = train_student(s1, teacher, tr, va, gold_va, args,
                                    device, ctx, True, "distilled")
    f_hd, e_hd, _ = train_student(s2, teacher, tr, va, gold_va, args,
                                  device, ctx, False, "hard_only")

    print(f"\n{'=' * 66}")
    print("summary")
    print(f"{'=' * 66}")

    def timed(m):
        ids = torch.randint(0, 32000, (8, 1024), device=device)
        msk = torch.ones(8, 1024, dtype=torch.long, device=device)
        pos = torch.rand(8, 1024, device=device)
        with torch.no_grad(), ctx:
            for _ in range(3):
                m(ids, attn_mask=msk, doc_pos=pos)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                m(ids, attn_mask=msk, doc_pos=pos)
            if device == "cuda":
                torch.cuda.synchronize()
        return (time.time() - t0) / 10 * 1000

    t_ms, s_ms = timed(teacher), timed(s1)
    print(f"  {'model':<22} {'params':>13} {'macro-F1':>9} {'ms/batch':>10}")
    print(f"  {'teacher (12 layer)':<22} {n_teacher:>13,} {tf:>9.4f} "
          f"{t_ms:>10.1f}")
    print(f"  {'student, distilled':<22} {n_s:>13,} {f_kd:>9.4f} {s_ms:>10.1f}")
    print(f"  {'student, hard only':<22} {n_s:>13,} {f_hd:>9.4f} {s_ms:>10.1f}")
    print(f"\n  distillation effect: {f_kd - f_hd:+.4f}")
    print(f"  retained vs teacher: {100*f_kd/tf:.1f}% of macro-F1 at "
          f"{100*n_s/n_teacher:.0f}% of parameters, {t_ms/s_ms:.2f}x faster")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"teacher_f1": tf, "teacher_params": n_teacher,
                   "student_params": n_s, "distilled_f1": f_kd,
                   "hard_only_f1": f_hd, "teacher_ms": t_ms,
                   "student_ms": s_ms, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()