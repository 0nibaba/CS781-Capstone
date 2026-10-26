"""Knowledge distillation for abstractive summarization (compulsory).

Teacher: summ_lorma (prefix-LM + LoRMA, dev BERTScore 0.8290 under
greedy + no-repeat-4). Student: the same architecture at 6 of 12 layers,
initialised from every other teacher layer, trained fully.

Soft target is the teacher's full vocabulary distribution at each summary
position -- 32,000 graded values per token, by far the densest supervision of
any phase. Two arms from one initialisation, as in Phases 3-5.

Cross-phase series of the KD term's contribution:
  Ph2 NER  70 docs    +0.0036      Ph4 CJPE 37,387  +0.0354
  Ph3 RR   80 docs    +0.0050      Ph5 LSI  42,750  +0.0083
Current hypothesis: the effect tracks how much the truncation COSTS, given
enough data to exploit softer targets. Generation is a far harder target than
classification, so the half-depth student should lose more here than LSI's did
(-0.0057), and the hypothesis therefore predicts a LARGER effect than +0.0083.
Registered before running.
"""
import os, copy, json, argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from bert_score import BERTScorer

from summ_data import SummPairs, load_split, batches, collate, TOKENIZER
from summ_model import Summarizer, strip_eos
from train_summ import get_lr
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_student(teacher, keep_every=2):
    s = copy.deepcopy(teacher)
    if s.arch == "encdec":
        blocks = s.base.decoder
        kept = [blocks[i] for i in range(0, len(blocks), keep_every)]
        s.base.decoder = nn.ModuleList(kept)
        s.base.config.n_decoder_layer = len(kept)
    else:
        blocks = s.base.transformer.h
        kept = [blocks[i] for i in range(0, len(blocks), keep_every)]
        s.base.transformer.h = nn.ModuleList(kept)
        s.base.config.n_layer = len(kept)
    for q in s.parameters():
        q.requires_grad = True
    return s


@torch.no_grad()
def bert_eval(model, docs, tok, device, ctx, arch, scorer, bs=8):
    model.eval()
    ds = SummPairs(docs, tok, with_target=False)
    first = [it for it in ds.items if it["chunk"] == 0]
    cands = []
    for i in range(0, len(first), bs):
        b = collate(first[i:i + bs], device, arch)
        with ctx:
            seq = model.generate(b, max_len=544, min_len=480,
                                 temperature=0.0, no_repeat_ngram=4)
        cands.extend(strip_eos(seq, tok))
    F_ = scorer.score(cands, [d["summary"] for d in docs])[2].mean().item()
    model.train()
    return F_


def train_arm(student, teacher, tr, dv_docs, tok, args, device, ctx, scorer,
              distil, tag):
    print(f"\n{'-'*66}\n{tag}: {'WITH' if distil else 'WITHOUT'} distillation"
          f"\n{'-'*66}")
    params = [q for q in student.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    nodecay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95))

    n_tr = sum(1 for it in tr.items if it["chunk"] == 0)
    spe = max(1, -(-n_tr // args.batch_size))
    total = spe * args.epochs
    warmup = max(1, int(0.06 * total))
    best, step = -1.0, 0
    run_h, run_k, nb = 0.0, 0.0, 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        for c, b in batches(tr, args.batch_size, device, student.arch,
                            seed=args.seed + epoch, chunk0_only=True):
            lr = get_lr(step, warmup, total, args.max_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                s_logits, hard = student(b)
                loss, kd = hard, torch.tensor(0.0, device=device)
                if distil:
                    with torch.no_grad():
                        t_logits, _ = teacher(b)
                    if student.arch == "encdec":
                        mask = b["dec_mask"].bool().reshape(-1)
                        sl = s_logits.reshape(-1, s_logits.size(-1))[mask]
                        tl = t_logits.reshape(-1, t_logits.size(-1))[mask]
                    else:
                        mask = (b["targets"][:, 1:] != -100).reshape(-1)
                        sl = s_logits[:, :-1].reshape(
                            -1, s_logits.size(-1))[mask]
                        tl = t_logits[:, :-1].reshape(
                            -1, t_logits.size(-1))[mask]
                    T = args.temperature
                    kd = F.kl_div(F.log_softmax(sl.float() / T, -1),
                                  F.softmax(tl.float() / T, -1),
                                  reduction="batchmean") * (T * T)
                    loss = (1 - args.alpha) * hard + args.alpha * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_h += hard.item(); run_k += float(kd.detach()); nb += 1
            step += 1

        bf = bert_eval(student, dv_docs[:args.gen_n], tok, device, ctx,
                       student.arch, scorer)
        flag = "  <- best" if bf > best else ""
        print(f"  ep {epoch} | hard {run_h/nb:.4f} | kd {run_k/nb:.4f} "
              f"| BERTScore {bf:.4f}{flag}")
        run_h, run_k, nb = 0.0, 0.0, 0
        if bf > best:
            best = bf
            torch.save({"model": student.state_dict(), "arch": student.arch,
                        "epoch": epoch, "bert_f1": bf,
                        "keep_every": args.keep_every, "distilled": distil,
                        "args": vars(args)},
                       os.path.join(args.out_dir, f"{tag}_best.pt"))
    print(f"  best {best:.4f}")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--run_name", default="summ_distil")
    p.add_argument("--keep_every", type=int, default=2)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_n", type=int, default=50)
    p.add_argument("--max_lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1899)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    args.out_dir = os.path.join(ROOT, "checkpoints/summ", args.run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(os.path.join(args.teacher_dir, "ckpt_best.pt"),
                    map_location="cpu", weights_only=False)
    t_args = ck.get("args", {})
    base, arch, d_model = load_base(args.base_checkpoint, device)
    teacher = Summarizer(base, arch)
    if t_args.get("finetune") == "lorma":
        from lorma import apply_lorma
        apply_lorma(teacher.base, r=t_args.get("lorma_r", 8),
                    alpha=t_args.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
    teacher.load_state_dict(ck["model"], strict=True)
    teacher = teacher.to(device).eval()
    for q in teacher.parameters():
        q.requires_grad = False
    n_t = sum(q.numel() for q in teacher.parameters())
    print(f"teacher: {arch}, {n_t:,} params, "
          f"train-time BERTScore {ck.get('bert_f1', 0):.4f}")

    tok = Tokenizer.from_file(TOKENIZER)
    tr = SummPairs(load_split("train"), tok)
    dv_docs = load_split("dev")
    scorer = BERTScorer(lang="en", rescale_with_baseline=False)

    s1 = make_student(teacher, args.keep_every).to(device)
    s2 = copy.deepcopy(s1).to(device)
    n_s = sum(q.numel() for q in s1.parameters())
    print(f"student: {n_s:,} params ({100*n_s/n_t:.1f}%)")
    print("baseline: lead-512 extraction = 0.8300\n")

    f_kd = train_arm(s1, teacher, tr, dv_docs, tok, args, device, ctx, scorer,
                     True, "distilled")
    f_hd = train_arm(s2, teacher, tr, dv_docs, tok, args, device, ctx, scorer,
                     False, "hard_only")

    tf = ck.get("bert_f1", 0)
    print(f"\n{'='*66}")
    print(f"  teacher (train-time) {tf:.4f}")
    print(f"  student, distilled   {f_kd:.4f}")
    print(f"  student, hard only   {f_hd:.4f}")
    print(f"  distillation effect: {f_kd - f_hd:+.4f}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"teacher_f1": tf, "distilled_f1": f_kd,
                   "hard_only_f1": f_hd, "teacher_params": n_t,
                   "student_params": n_s, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()