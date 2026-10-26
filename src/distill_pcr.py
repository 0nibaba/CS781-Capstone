"""Knowledge distillation for PCR (compulsory).

Teacher: pcr_full (decoder, full fine-tune, dev F1@7 0.3346).
Student: same architecture at 6 of 12 layers, initialised from every other
teacher layer.

Retrieval-specific signal: the student matches the teacher's SIMILARITY
DISTRIBUTION over each batch -- KL between softmax(teacher q.c / T) and the
student's, which is the ranking signal rather than a class posterior. Combined
with the student's own InfoNCE against the gold positive.

Cross-phase series of the KD term's contribution, and the two factors that
have predicted it since Phase 4 (how much truncation costs x enough data):
  Ph3 RR   cost 0.0710 / 80 docs      +0.0050
  Ph4 CJPE cost 0.0567 / 37,387       +0.0354
  Ph5 LSI  cost 0.0057 / 42,750       +0.0083
  Ph6 Summ cost 0.0017 / 6,528        +0.0004
Here there are only 5,308 training pairs, so the data factor is weak. Registered
prediction: the effect is small (< +0.01) REGARDLESS of what halving the depth
costs, because there is not enough data to exploit a softer target.
"""
import os, copy, json, math, argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from pcr_baseline import load_split, ceiling, TOKENIZER
from pcr_data import Corpus, PairSampler
from pcr_model import CaseEncoder
from train_pcr import get_lr, rank_dev
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_student(teacher, keep_every=2):
    s = copy.deepcopy(teacher)
    blocks = s.base.transformer.h
    kept = [blocks[i] for i in range(0, len(blocks), keep_every)]
    s.base.transformer.h = nn.ModuleList(kept)
    s.base.config.n_layer = len(kept)
    for q in s.parameters():
        q.requires_grad = True
    return s


def train_arm(student, teacher, sampler, dvq, dv_qc, dv_cc, args, device,
              ctx, distil, tag):
    print(f"\n{'-'*66}\n{tag}: {'WITH' if distil else 'WITHOUT'} distillation"
          f"\n{'-'*66}")
    params = [q for q in student.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    spe = max(1, len(sampler) // args.batch_size)
    total = spe * args.epochs
    warmup = max(1, int(0.1 * total))
    best, step = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        run_h, run_k, nb = 0.0, 0.0, 0
        for keep, (qi, qm), (ci, cm), _ in sampler.batches(
                args.batch_size, device, epoch=epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                qv = student.embed(qi, qm)
                cv = student.embed(ci, cm)
                hard = student.info_nce(qv, cv, temp=args.temp)
                loss, kd = hard, torch.tensor(0.0, device=device)
                if distil:
                    with torch.no_grad():
                        tq = teacher.embed(qi, qm)
                        tc = teacher.embed(ci, cm)
                        t_sim = (tq @ tc.T) / args.temperature
                    s_sim = (qv @ cv.T) / args.temperature
                    kd = F.kl_div(F.log_softmax(s_sim.float(), -1),
                                  F.softmax(t_sim.float(), -1),
                                  reduction="batchmean")
                    loss = (1 - args.alpha) * hard + args.alpha * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_h += hard.item(); run_k += float(kd.detach()); nb += 1
            step += 1

        (f1, pr, rc), _ = rank_dev(student, dv_qc, dv_cc, dvq, device, ctx)
        flag = "  <- best" if f1 > best else ""
        print(f"  epoch {epoch} | hard {run_h/nb:.4f} | kd {run_k/nb:.4f} "
              f"| F1@7 {f1:.4f}{flag}")
        if f1 > best:
            best = f1
            torch.save({"model": student.state_dict(), "arch": student.arch,
                        "epoch": epoch, "f1": f1, "keep_every": args.keep_every,
                        "distilled": distil, "args": vars(args)},
                       os.path.join(args.out_dir, f"{tag}_best.pt"))
    print(f"  best {best:.4f}")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--run_name", default="pcr_distil")
    p.add_argument("--keep_every", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
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
    args.out_dir = os.path.join(ROOT, "checkpoints/pcr", args.run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(os.path.join(args.teacher_dir, "ckpt_best.pt"),
                    map_location="cpu", weights_only=False)
    t_args = ck.get("args", {})
    base, arch, d_model = load_base(args.base_checkpoint, device)
    teacher = CaseEncoder(base, arch, d_model,
                          pooling=t_args.get("pooling", "mean"))
    teacher.embed_dim = d_model
    teacher.load_state_dict(ck["model"], strict=True)
    teacher = teacher.to(device).eval()
    for q in teacher.parameters():
        q.requires_grad = False
    n_t = sum(q.numel() for q in teacher.parameters())
    print(f"teacher: {arch}, {n_t:,} params, dev F1@7 {ck.get('f1', 0):.4f}")

    tok = Tokenizer.from_file(TOKENIZER)
    trq, trc = load_split("train")
    dvq, dvc = load_split("dev")
    tr_qc = Corpus({k: v["text"] for k, v in trq.items()}, tok)
    tr_cc = Corpus(trc, tok)
    dv_qc = Corpus({k: v["text"] for k, v in dvq.items()}, tok)
    dv_cc = Corpus(dvc, tok)
    sampler = PairSampler(trq, tr_qc, tr_cc, seed=args.seed)
    print(f"{len(sampler)} pairs | dev ceiling {ceiling(dvq)[0]:.4f} | "
          f"BM25@4096 0.2991")

    s1 = make_student(teacher, args.keep_every).to(device)
    s1.embed_dim = d_model
    s2 = copy.deepcopy(s1).to(device)
    s2.embed_dim = d_model
    n_s = sum(q.numel() for q in s1.parameters())
    print(f"student: {n_s:,} params ({100*n_s/n_t:.1f}%)")

    f_kd = train_arm(s1, teacher, sampler, dvq, dv_qc, dv_cc, args, device,
                     ctx, True, "distilled")
    f_hd = train_arm(s2, teacher, sampler, dvq, dv_qc, dv_cc, args, device,
                     ctx, False, "hard_only")

    tf = ck.get("f1", 0)
    print(f"\n{'='*66}")
    print(f"  teacher            {tf:.4f}")
    print(f"  student, distilled {f_kd:.4f}")
    print(f"  student, hard only {f_hd:.4f}")
    print(f"  distillation effect: {f_kd - f_hd:+.4f}")
    print(f"  truncation cost:     {tf - f_hd:+.4f}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"teacher_f1": tf, "distilled_f1": f_kd,
                   "hard_only_f1": f_hd, "teacher_params": n_t,
                   "student_params": n_s, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()