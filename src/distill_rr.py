import os
import copy
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from rr_data import (RRBlocks, load_split, batches, TOKENIZER,
                     N_LABELS, IGNORE)
from rr_model import SentenceClassifier
from ner_model import load_base
from train_rr import get_lr, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_student(teacher, keep_every=2):
    student = copy.deepcopy(teacher)
    blocks = student.base.transformer.h
    kept = [blocks[i] for i in range(0, len(blocks), keep_every)]
    student.base.transformer.h = nn.ModuleList(kept)
    student.base.config.n_layer = len(kept)
    return student


def train_student(student, teacher, tr, va, dev_docs, args, device, ctx,
                  distil, tag):
    print(f"\n{'-' * 66}")
    print(f"{tag}: {'WITH' if distil else 'WITHOUT'} distillation")
    print(f"{'-' * 66}")

    params = [q for q in student.parameters() if q.requires_grad]
    n = sum(q.numel() for q in params)

    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
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

        for chunk, (ids, msk, pos, pmask, lab) in batches(
                tr, args.batch_size, device, seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)

            with ctx:
                s_logits, hard = student(ids, msk, pos, pmask, labels=lab)
                loss = hard
                kd = torch.tensor(0.0, device=device)

                if distil:
                    with torch.no_grad():
                        t_logits, _ = teacher(ids, msk, pos, pmask)
                    T = args.temperature
                    valid = (lab != IGNORE).view(-1)
                    s = F.log_softmax(
                        s_logits.float().view(-1, N_LABELS)[valid] / T, -1)
                    t = F.softmax(
                        t_logits.float().view(-1, N_LABELS)[valid] / T, -1)
                    # T^2 keeps the KD gradient comparable to the hard loss
                    kd = F.kl_div(s, t, reduction="batchmean") * (T * T)
                    loss = (1 - args.alpha) * hard + args.alpha * kd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            ep_h += hard.item()
            ep_k += float(kd)
            nb += 1
            step += 1

        mf, mp, mr, acc, per, by_dom, pred = evaluate(
            student, va, dev_docs, args.batch_size, device, ctx)
        flag = "  <- best" if mf > best else ""
        print(f"  epoch {epoch:3d} | hard {ep_h/nb:7.4f} | kd {ep_k/nb:7.4f} "
              f"| macro-F1 {mf:.4f} | acc {acc:.4f}{flag}")
        if mf > best:
            best, best_ep = mf, epoch
            torch.save({"model": student.state_dict(), "arch": args.arch,
                        "epoch": epoch, "macro_f1": mf, "args": vars(args),
                        "distilled": distil, "keep_every": args.keep_every},
                       os.path.join(args.out_dir, f"{tag}_best.pt"))

    print(f"  best {best:.4f} at epoch {best_ep}")
    return best, best_ep, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True,
                   help="checkpoints/rr/<run> holding the fine-tuned teacher")
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--keep_every", type=int, default=2, help="2 keeps 6 of 12")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1899)
    p.add_argument("--run_name", default="rr_distil")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    args.out_dir = os.path.join(ROOT, "checkpoints/rr", args.run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    ck_path = os.path.join(args.teacher_dir, "ckpt_best.pt")
    if not os.path.exists(ck_path):
        ck_path = os.path.join(args.teacher_dir, "ckpt_final.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    t_args = ck.get("args", {})
    n_ctx = t_args.get("context", 8)
    frac = t_args.get("ctx_frac", 0.34)
    mx = t_args.get("max_len", 1024)
    has_crf = any(k.startswith("crf.") for k in ck["model"])
    print(f"teacher geometry: context={n_ctx} ctx_frac={frac} max_len={mx} "
          f"crf={has_crf}")

    tok = Tokenizer.from_file(TOKENIZER)
    train_docs = load_split("train", ("CL", "IT"))
    dev_docs = load_split("dev", ("CL", "IT"))
    tr = RRBlocks(train_docs, tok, max_len=mx, context=n_ctx, ctx_frac=frac)
    va = RRBlocks(dev_docs, tok, max_len=mx, context=n_ctx, ctx_frac=frac)
    print(f"train {len(train_docs)} docs -> {len(tr)} blocks | "
          f"dev {len(dev_docs)} -> {len(va)}")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    args.arch = arch
    teacher = SentenceClassifier(base, arch, d_model, dropout=0.0,
                                 use_crf=has_crf)
    teacher.load_state_dict(ck["model"], strict=True)
    teacher = teacher.to(device).eval()
    for q in teacher.parameters():
        q.requires_grad = False

    n_teacher = sum(q.numel() for q in teacher.parameters())
    tf, _, _, tacc, _, _, _ = evaluate(teacher, va, dev_docs,
                                       args.batch_size, device, ctx)
    print(f"teacher: {n_teacher:,} params, "
          f"{len(teacher.base.transformer.h)} layers, "
          f"dev macro-F1 {tf:.4f} / acc {tacc:.4f}")

    s1 = make_student(teacher, args.keep_every).to(device)
    s2 = copy.deepcopy(s1).to(device)
    for q in list(s1.parameters()) + list(s2.parameters()):
        q.requires_grad = True
    n_s = sum(q.numel() for q in s1.parameters())
    print(f"student: {n_s:,} params, {len(s1.base.transformer.h)} layers "
          f"({100*n_s/n_teacher:.1f}% of the teacher)")

    f_kd, e_kd, _ = train_student(s1, teacher, tr, va, dev_docs, args,
                                  device, ctx, True, "distilled")
    f_hd, e_hd, _ = train_student(s2, teacher, tr, va, dev_docs, args,
                                  device, ctx, False, "hard_only")

    def timed(m):
        b, L, S = 4, mx, 32
        ids = torch.randint(0, 32000, (b, L), device=device)
        msk = torch.ones(b, L, dtype=torch.long, device=device)
        pos = torch.randint(0, L, (b, S), device=device)
        pm = torch.ones(b, S, dtype=torch.long, device=device)
        with torch.no_grad(), ctx:
            for _ in range(3):
                m(ids, msk, pos, pm)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                m(ids, msk, pos, pm)
            if device == "cuda":
                torch.cuda.synchronize()
        return (time.time() - t0) / 10 * 1000

    t_ms, s_ms = timed(teacher), timed(s1)

    print(f"\n{'=' * 66}\nsummary\n{'=' * 66}")
    print(f"  {'model':<22} {'params':>13} {'macro-F1':>9} {'ms/batch':>10}")
    print(f"  {'teacher (12 layer)':<22} {n_teacher:>13,} {tf:>9.4f} {t_ms:>10.1f}")
    print(f"  {'student, distilled':<22} {n_s:>13,} {f_kd:>9.4f} {s_ms:>10.1f}")
    print(f"  {'student, hard only':<22} {n_s:>13,} {f_hd:>9.4f} {s_ms:>10.1f}")
    print(f"\n  distillation effect: {f_kd - f_hd:+.4f}")
    print(f"  retained vs teacher: {100*f_kd/tf:.1f}% of macro-F1 at "
          f"{100*n_s/n_teacher:.0f}% of parameters, {t_ms/s_ms:.2f}x faster")
    print(f"\n  dev is 10 documents -- treat |effect| < 0.02 as a null.")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"teacher_f1": tf, "teacher_params": n_teacher,
                   "student_params": n_s, "distilled_f1": f_kd,
                   "hard_only_f1": f_hd, "distilled_epoch": e_kd,
                   "hard_only_epoch": e_hd, "teacher_ms": t_ms,
                   "student_ms": s_ms, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()