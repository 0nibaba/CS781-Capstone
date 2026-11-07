import os, json, argparse, zipfile, datetime

import numpy as np
import torch
from tokenizers import Tokenizer

from pcr_baseline import load_split, TOKENIZER
from pcr_data import Corpus
from pcr_model import CaseEncoder
from train_pcr import embed_all
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--note", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = CaseEncoder(base, arch, d_model, pooling=tr.get("pooling", "mean"))
    model.embed_dim = d_model
    if tr.get("finetune") == "lorma" or any(".c_attn.A." in k
                                            for k in ck["model"]):
        from lorma import apply_lorma
        apply_lorma(model.base, r=tr.get("lorma_r", 8),
                    alpha=tr.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
    if "keep_every" in ck:
        ke = ck["keep_every"]
        b_ = model.base.transformer.h
        model.base.transformer.h = torch.nn.ModuleList(
            [b_[i] for i in range(0, len(b_), ke)])
        model.base.config.n_layer = len(model.base.transformer.h)
        print(f"distilled student: {model.base.config.n_layer} layers")
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (list(missing)[:5],
                                            list(unexpected)[:5])
    model = model.to(device).eval()
    print(f"arch {arch} | finetune {tr.get('finetune')} | "
          f"dev F1@7 {ck.get('f1', float('nan')):.4f}")

    q, c = load_split("test")
    tok = Tokenizer.from_file(args.tokenizer)
    qcorp = Corpus({k: v["text"] for k, v in q.items()}, tok)
    ccorp = Corpus(c, tok)
    print(f"test {len(q)} queries, {len(c)} candidates")

    cflat = ccorp.flat()
    cpos = {k: i for i, k in enumerate(ccorp.ids)}
    cowner = torch.tensor([cpos[k] for k, _ in cflat])
    CE = embed_all(model, [x for _, x in cflat], device, ctx)

    out = {}
    with torch.no_grad():
        for qid in qcorp.ids:
            QE = embed_all(model, qcorp.chunks[qid], device, ctx)
            best = (QE @ CE.T).max(0).values
            per_doc = torch.full((len(ccorp.ids),), -1e9).scatter_reduce(
                0, cowner, best, reduce="amax", include_self=True)
            top = torch.argsort(-per_doc)[:args.k].tolist()
            out[qid] = [ccorp.ids[i] for i in top]

    pool = set(ccorp.ids)
    for qid, v in out.items():
        assert len(v) == args.k and len(set(v)) == args.k
        assert all(x in pool for x in v)
    assert len(out) == len(q)

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    jp = os.path.join(sub_dir, "prediction.json")
    with open(jp, "w") as f:
        json.dump(out, f)
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          7 (PCR)\n")
        f.write(f"checkpoint     {args.checkpoint}\n")
        f.write(f"arch {arch} | finetune {tr.get('finetune')} | "
                f"pooling {tr.get('pooling')} | hard_neg "
                f"{tr.get('hard_negatives')}\n")
        f.write(f"dev F1@7       {ck.get('f1', float('nan')):.4f} "
                f"(BM25@4096 0.2991, ceiling 0.6181)\n")
        if args.note:
            f.write(f"note           {args.note}\n")
        for k_, v_ in sorted(tr.items()):
            f.write(f"  {k_:18s} {v_}\n")
    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")
    print(f"\n{sub_dir}/ written — {len(out)} queries x {args.k}")


if __name__ == "__main__":
    main()