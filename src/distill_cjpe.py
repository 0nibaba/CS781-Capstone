import os, copy, json, time, argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from cjpe_data import CJPEDocs, load_split, batches, TOKENIZER
from cjpe_model import DocClassifier
from ner_model import load_base
from train_cjpe import get_lr, evaluate, macro_f1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PRIOR = torch.tensor([22067 / 37387, 15320 / 37387])


def make_student(teacher, keep_every=2):
    student = copy.deepcopy(teacher)
    blocks = student.base.transformer.h
    kept = [blocks[i] for i in range(0, len(blocks), keep_every)]
    student.base.transformer.h = nn.ModuleList(kept)
    student.base.config.n_layer = len(kept)
    return student


def train_arm(student, teacher, tr, va, args, device, ctx, distil, tag):
    print(f"\n{'-'*66}\n{tag}: {'WITH' if distil else 'WITHOUT'} distillation"
          f"\n{'-'*66}")
    log_prior = TRAIN_PRIOR.log()
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

    best, best_tag, step = -1.0, "raw", 0
    run_h, run_k, nb = 0.0, 0.0, 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        for chunk, (ids, msk, lab) in batches(tr, args.batch_size, device,
                                              seed=args.seed + epoch):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                s_logits, hard = student(ids, msk, labels=lab)
                loss, kd = hard, torch.tensor(0.0, device=device)
                if distil:
                    with torch.no_grad():
                        t_logits, _ = teacher(ids, msk)
                    T = args.temperature
                    s = F.log_softmax(s_logits.float() / T, -1)
                    t = F.softmax(t_logits.float() / T, -1)
                    kd = F.kl_div(s, t, reduction="batchmean") * (T * T)
                    loss = (1 - args.alpha) * hard + args.alpha * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_h += hard.item(); run_k += float(kd.detach()); nb += 1
            step += 1

            if step % args.eval_every == 0:
                mf, mf_adj, acc, f1s, p1, _, _ = evaluate(
                    student, va, args.batch_size, device, ctx, log_prior)
                score, rule = max(mf, mf_adj), ("adj" if mf_adj > mf else "raw")
                flag = "  <- best" if score > best else ""
                print(f"  ep {epoch} step {step:5d} | hard {run_h/nb:.4f} "
                      f"| kd {run_k/nb:.4f} | macro-F1 {mf:.4f} "
                      f"| +prior {mf_adj:.4f} | pred1 {p1:.3f}{flag}")
                run_h, run_k, nb = 0.0, 0.0, 0
                if score > best:
                    best, best_tag = score, rule
                    torch.save({"model": student.state_dict(),
                                "arch": args.arch, "step": step,
                                "epoch": epoch, "macro_f1": score,
                                "decision": rule, "distilled": distil,
                                "keep_every": args.keep_every,
                                "args": vars(args)},
                               os.path.join(args.out_dir, f"{tag}_best.pt"))
    print(f"  best {best:.4f} ({best_tag})")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--run_name", default="cjpe_distil")
    p.add_argument("--keep_every", type=int, default=2)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=1500)
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
    args.out_dir = os.path.join(ROOT, "checkpoints/cjpe", args.run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(os.path.join(args.teacher_dir, "ckpt_best.pt"),
                    map_location="cpu", weights_only=False)
    t_args = ck.get("args", {})
    args.mode = t_args.get("mode", "tail")
    args.pooling = t_args.get("pooling", "last")
    args.max_len = t_args.get("max_len", 1024)
    print(f"teacher geometry: mode={args.mode} pooling={args.pooling} "
          f"max_len={args.max_len}")

    tok = Tokenizer.from_file(TOKENIZER)
    tr = CJPEDocs(load_split("train"), tok, mode=args.mode,
                  max_len=args.max_len)
    va = CJPEDocs(load_split("dev"), tok, mode=args.mode,
                  max_len=args.max_len)
    print(f"train {len(tr)} | dev {len(va)}")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    args.arch = arch
    teacher = DocClassifier(base, arch, d_model, dropout=0.0,
                            pooling=args.pooling)
    teacher.load_state_dict(ck["model"], strict=True)
    teacher = teacher.to(device).eval()
    for q in teacher.parameters():
        q.requires_grad = False

    n_t = sum(q.numel() for q in teacher.parameters())
    tf, tf_adj, tacc, _, _, _, _ = evaluate(teacher, va, args.batch_size,
                                            device, ctx, TRAIN_PRIOR.log())
    print(f"teacher: {n_t:,} params, {len(teacher.base.transformer.h)} layers,"
          f" dev {max(tf, tf_adj):.4f}")

    s1 = make_student(teacher, args.keep_every).to(device)
    s2 = copy.deepcopy(s1).to(device)
    for q in list(s1.parameters()) + list(s2.parameters()):
        q.requires_grad = True
    n_s = sum(q.numel() for q in s1.parameters())
    print(f"student: {n_s:,} params ({100*n_s/n_t:.1f}%), "
          f"{len(s1.base.transformer.h)} layers")

    f_kd = train_arm(s1, teacher, tr, va, args, device, ctx, True, "distilled")
    f_hd = train_arm(s2, teacher, tr, va, args, device, ctx, False, "hard_only")

    print(f"\n{'='*66}\n  teacher            {max(tf,tf_adj):.4f}")
    print(f"  student, distilled {f_kd:.4f}")
    print(f"  student, hard only {f_hd:.4f}")
    print(f"  distillation effect: {f_kd - f_hd:+.4f}")
    print(f"  retained: {100*f_kd/max(tf,tf_adj):.1f}% at "
          f"{100*n_s/n_t:.0f}% of parameters")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"teacher_f1": max(tf, tf_adj), "distilled_f1": f_kd,
                   "hard_only_f1": f_hd, "teacher_params": n_t,
                   "student_params": n_s, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()