"""Knowledge distillation for LSI (compulsory).

Teacher: lsi_full3 (decoder, full fine-tune, dev 0.4004).
Student: every other layer (6/12), head copied.

Multi-label distillation is per-label binary: the soft target is the teacher's
sigmoid probability for each of the 100 statutes, matched with BCE. That gives
the student 100 graded targets per document rather than CJPE's single one --
so if the Phase 4 account is right (the KL term is a regulariser whose value
scales with the density of supervision), the effect here should be at least as
large as CJPE's +0.0354. Registered before running.
"""
import os, copy, json, argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from lsi_data import LSIDocs, load_split, batches, TOKENIZER, N_LABELS
from lsi_model import StatuteClassifier
from lsi_metrics import scores, best_global, best_per_label
from train_lsi import get_lr, predict
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_student(teacher, keep_every=2):
    s = copy.deepcopy(teacher)
    blocks = s.base.transformer.h
    kept = [blocks[i] for i in range(0, len(blocks), keep_every)]
    s.base.transformer.h = nn.ModuleList(kept)
    s.base.config.n_layer = len(kept)
    return s


def train_arm(student, teacher, tr, va, args, device, ctx, distil, tag):
    print(f"\n{'-'*66}\n{tag}: {'WITH' if distil else 'WITHOUT'} distillation"
          f"\n{'-'*66}")
    params = [q for q in student.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    spe = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total = spe * args.epochs
    warmup = max(1, int(0.06 * total))
    best, step = -1.0, 0
    run_h, run_k, nb = 0.0, 0.0, 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        for chunk, (ids, msk, y) in batches(tr, args.batch_size, device,
                                            seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                s_logits, hard = student(ids, msk, labels=y)
                loss, kd = hard, torch.tensor(0.0, device=device)
                if distil:
                    with torch.no_grad():
                        t_logits, _ = teacher(ids, msk)
                    T = args.temperature
                    soft = torch.sigmoid(t_logits.float() / T)
                    kd = F.binary_cross_entropy_with_logits(
                        s_logits.float() / T, soft) * (T * T)
                    loss = (1 - args.alpha) * hard + args.alpha * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_h += hard.item(); run_k += float(kd.detach()); nb += 1
            step += 1

            if step % args.eval_every == 0:
                prob, gold = predict(student, va, args.batch_size, device, ctx)
                m05 = scores(prob, gold, 0.5)[0]
                mpl, thr = best_per_label(prob, gold)
                flag = "  <- best" if mpl > best else ""
                print(f"  ep {epoch} step {step:5d} | hard {run_h/nb:.4f} "
                      f"| kd {run_k/nb:.4f} | macro@.5 {m05:.4f} "
                      f"| per-label {mpl:.4f}{flag}")
                run_h, run_k, nb = 0.0, 0.0, 0
                if mpl > best:
                    best = mpl
                    torch.save({"model": student.state_dict(),
                                "arch": args.arch, "step": step,
                                "epoch": epoch, "macro_f1": mpl,
                                "macro_05": m05,
                                "thresholds": thr.tolist(),
                                "thr_global": best_global(prob, gold)[1],
                                "macro_global": best_global(prob, gold)[0],
                                "keep_every": args.keep_every,
                                "distilled": distil, "args": vars(args)},
                               os.path.join(args.out_dir, f"{tag}_best.pt"))
    print(f"  best {best:.4f}")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--run_name", default="lsi_distil")
    p.add_argument("--keep_every", type=int, default=2)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--max_lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1899)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    args.out_dir = os.path.join(ROOT, "checkpoints/lsi", args.run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(os.path.join(args.teacher_dir, "ckpt_best.pt"),
                    map_location="cpu", weights_only=False)
    t_args = ck.get("args", {})
    args.mode = t_args.get("mode", "head")
    args.pooling = t_args.get("pooling", "mean")
    args.max_len = t_args.get("max_len", 1024)

    tok = Tokenizer.from_file(TOKENIZER)
    tr = LSIDocs(load_split("train"), tok, mode=args.mode,
                 max_len=args.max_len)
    va = LSIDocs(load_split("dev"), tok, mode=args.mode, max_len=args.max_len)
    print(f"train {len(tr)} | dev {len(va)} | mode {args.mode}")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    args.arch = arch
    teacher = StatuteClassifier(base, arch, d_model, dropout=0.0,
                                pooling=args.pooling)
    teacher.load_state_dict(ck["model"], strict=True)
    teacher = teacher.to(device).eval()
    for q in teacher.parameters():
        q.requires_grad = False
    n_t = sum(q.numel() for q in teacher.parameters())
    print(f"teacher: {n_t:,} params, dev {ck.get('macro_f1', 0):.4f}")

    s1 = make_student(teacher, args.keep_every).to(device)
    s2 = copy.deepcopy(s1).to(device)
    for q in list(s1.parameters()) + list(s2.parameters()):
        q.requires_grad = True
    n_s = sum(q.numel() for q in s1.parameters())
    print(f"student: {n_s:,} params ({100*n_s/n_t:.1f}%), "
          f"{len(s1.base.transformer.h)} layers")

    f_kd = train_arm(s1, teacher, tr, va, args, device, ctx, True, "distilled")
    f_hd = train_arm(s2, teacher, tr, va, args, device, ctx, False, "hard_only")

    print(f"\n{'='*66}")
    print(f"  teacher            {ck.get('macro_f1', 0):.4f}")
    print(f"  student, distilled {f_kd:.4f}")
    print(f"  student, hard only {f_hd:.4f}")
    print(f"  distillation effect: {f_kd - f_hd:+.4f}")
    print(f"  retained: {100*f_kd/ck.get('macro_f1', 1):.1f}% at "
          f"{100*n_s/n_t:.0f}% of parameters")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"teacher_f1": ck.get("macro_f1"), "distilled_f1": f_kd,
                   "hard_only_f1": f_hd, "teacher_params": n_t,
                   "student_params": n_s, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()