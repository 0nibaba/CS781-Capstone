import os, json, itertools, argparse, zipfile, datetime

import numpy as np
import torch
from tokenizers import Tokenizer

from pcr_baseline import (load_split, micro_f1_at_k, ceiling, bm25_rank,
                          TOKENIZER)
from pcr_data import Corpus
from pcr_model import CaseEncoder
from train_pcr import embed_all
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(ck_path, base_path, device):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    base, arch, d_model = load_base(base_path, device)
    m = CaseEncoder(base, arch, d_model, pooling=tr.get("pooling", "mean"))
    m.embed_dim = d_model
    if "keep_every" in ck:
        ke = ck["keep_every"]
        b_ = m.base.transformer.h
        m.base.transformer.h = torch.nn.ModuleList(
            [b_[i] for i in range(0, len(b_), ke)])
        m.base.config.n_layer = len(m.base.transformer.h)
    if tr.get("finetune") == "lorma" or any(".c_attn.A." in k
                                            for k in ck["model"]):
        from lorma import apply_lorma
        apply_lorma(m.base, r=tr.get("lorma_r", 8),
                    alpha=tr.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected
    return m.to(device).eval(), ck


@torch.no_grad()
def dense_rank(model, qcorp, ccorp, device, ctx):
    cflat = ccorp.flat()
    cpos = {k: i for i, k in enumerate(ccorp.ids)}
    cowner = torch.tensor([cpos[k] for k, _ in cflat])
    CE = embed_all(model, [x for _, x in cflat], device, ctx)
    out = {}
    for q in qcorp.ids:
        QE = embed_all(model, qcorp.chunks[q], device, ctx)
        best = (QE @ CE.T).max(0).values
        per = torch.full((len(ccorp.ids),), -1e9).scatter_reduce(
            0, cowner, best, reduce="amax", include_self=True)
        out[q] = [ccorp.ids[i] for i in torch.argsort(-per).tolist()]
    return out


def rrf(rankings, weights, k=60):
    """rankings: list of {qid: [cid ranked]}; returns fused {qid: [cid]}."""
    out = {}
    for q in rankings[0]:
        s = {}
        for r, w in zip(rankings, weights):
            if w == 0:
                continue
            for pos, c in enumerate(r[q]):
                s[c] = s.get(c, 0.0) + w / (k + pos + 1)
        out[q] = [c for c, _ in sorted(s.items(), key=lambda kv: -kv[1])]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="ckpt:base pairs")
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--bm25_budget", type=int, default=4096)
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--sub_name", default="")
    p.add_argument("--use_bm25", type=int, default=-1,
                   help="-1 pick on dev, 0 dense only, 1 force include")
    p.add_argument("--note", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    tok = Tokenizer.from_file(args.tokenizer)

    dvq, dvc = load_split("dev")
    dq = Corpus({k: v["text"] for k, v in dvq.items()}, tok)
    dc = Corpus(dvc, tok)
    print(f"dev {len(dvq)} queries | ceiling {ceiling(dvq)[0]:.4f}")

    names, dev_ranks, models = [], [], []
    for spec in args.models:
        ckp, basep = spec.split(":")
        m, ck = build(ckp, basep, device)
        nm = os.path.basename(os.path.dirname(ckp))
        r = dense_rank(m, dq, dc, device, ctx)
        print(f"  {nm:<16} dev F1@7 {micro_f1_at_k(r, dvq)[0]:.4f}")
        names.append(nm); dev_ranks.append(r); models.append((m, basep, ckp))

    bm = bm25_rank(dvq, dvc, tok, max_tokens=args.bm25_budget)
    print(f"  {'BM25@' + str(args.bm25_budget):<16} dev F1@7 "
          f"{micro_f1_at_k(bm, dvq)[0]:.4f}")

    n = len(dev_ranks)
    best_dense = (-1.0, None)
    best_all = (-1.0, None)
    grid = [0, 1, 2]
    for w in itertools.product(grid, repeat=n):
        if sum(w) == 0:
            continue
        f = micro_f1_at_k(rrf(dev_ranks, list(w), args.rrf_k), dvq)[0]
        if f > best_dense[0]:
            best_dense = (f, list(w) + [0])
        for wb in (1, 2, 3):
            f2 = micro_f1_at_k(
                rrf(dev_ranks + [bm], list(w) + [wb], args.rrf_k), dvq)[0]
            if f2 > best_all[0]:
                best_all = (f2, list(w) + [wb])

    print(f"\n  dense-only fusion      {best_dense[0]:.4f}  "
          f"weights {best_dense[1]}")
    print(f"  dense + BM25 fusion    {best_all[0]:.4f}  "
          f"weights {best_all[1]}")
    use_bm = (best_all[0] > best_dense[0]) if args.use_bm25 < 0 \
        else bool(args.use_bm25)
    score, weights = (best_all if use_bm else best_dense)
    print(f"  -> submitting {'dense+BM25' if use_bm else 'dense-only'} "
          f"at {score:.4f}")

    if not args.sub_name:
        return

    teq, tec = load_split("test")
    tq = Corpus({k: v["text"] for k, v in teq.items()}, tok)
    tc = Corpus(tec, tok)
    test_ranks = []
    for (m, basep, ckp), nm in zip(models, names):
        test_ranks.append(dense_rank(m, tq, tc, device, ctx))
        print(f"  ranked test with {nm}")
    test_ranks.append(bm25_rank(teq, tec, tok, max_tokens=args.bm25_budget))

    fused = rrf(test_ranks, weights, args.rrf_k)
    pool = set(tc.ids)
    out = {}
    for q, v in fused.items():
        top = v[:7]
        assert len(top) == 7 and len(set(top)) == 7
        assert all(x in pool for x in top)
        out[q] = top
    assert len(out) == len(teq)

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    jp = os.path.join(sub_dir, "prediction.json")
    with open(jp, "w") as f:
        json.dump(out, f)
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          7 (PCR)\n")
        f.write(f"method         reciprocal-rank fusion (k={args.rrf_k})\n")
        for nm, w in zip(names + [f"BM25@{args.bm25_budget}"], weights):
            f.write(f"  {nm:<18} weight {w}\n")
        f.write(f"dev F1@7       {score:.4f} "
                f"(best single dense 0.3346, BM25 0.2991, ceiling 0.6181)\n")
        if args.note:
            f.write(f"note           {args.note}\n")
    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")
    print(f"\n{sub_dir}/ written — {len(out)} queries x 7")

if __name__ == "__main__":
    main()