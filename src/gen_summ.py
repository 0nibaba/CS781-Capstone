import os, json, time, argparse, zipfile, datetime

import numpy as np
import torch
from tokenizers import Tokenizer
from bert_score import BERTScorer

from summ_data import SummPairs, load_split, collate, TOKENIZER
from summ_model import Summarizer, strip_eos
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = [
    # name,                min_len, max_len, temp, top_k, no_repeat
    ("greedy-480",             480,     544, 0.0,     0,   0),
    ("greedy-480-norep4",      480,     544, 0.0,     0,   4),
    ("greedy-256",             256,     544, 0.0,     0,   0),
    ("greedy-free",              0,     544, 0.0,     0,   0),
    ("topk50-480",             480,     544, 0.9,    50,   0),
    ("topk50-480-norep4",      480,     544, 0.9,    50,   4),
]


@torch.no_grad()
def run(model, docs, tok, device, ctx, arch, cfg, bs):
    _, min_len, max_len, temp, top_k, nrn = cfg
    ds = SummPairs(docs, tok, with_target=False)
    first = [it for it in ds.items if it["chunk"] == 0]
    out, lens = [], []
    for i in range(0, len(first), bs):
        b = collate(first[i:i + bs], device, arch)
        with ctx:
            seq = model.generate(b, max_len=max_len, min_len=min_len,
                                 temperature=temp, top_k=top_k,
                                 no_repeat_ngram=nrn)
        txt = strip_eos(seq, tok)
        out.extend(txt)
        lens.extend(len(tok.encode(t).ids) for t in txt)
    return out, float(np.mean(lens))


def rep_rate(text, n=4):
    w = text.split()
    if len(w) < n:
        return 0.0
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(g)) / len(g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sub_name", default="")
    p.add_argument("--config", default="", help="skip the sweep, use this one")
    p.add_argument("--note", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = Summarizer(base, arch)
    if "keep_every" in ck:
        ke = ck["keep_every"]
        if arch == "encdec":
            b_ = model.base.decoder
            model.base.decoder = torch.nn.ModuleList(
                [b_[i] for i in range(0, len(b_), ke)])
            model.base.config.n_decoder_layer = len(model.base.decoder)
        else:
            b_ = model.base.transformer.h
            model.base.transformer.h = torch.nn.ModuleList(
                [b_[i] for i in range(0, len(b_), ke)])
            model.base.config.n_layer = len(model.base.transformer.h)
        print(f"distilled student: {ke=} -> fewer layers")
    if tr.get("finetune") == "lorma" or any(".c_attn.A." in k
                                            for k in ck["model"]):
        from lorma import apply_lorma
        apply_lorma(model.base, r=tr.get("lorma_r", 8),
                    alpha=tr.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
    model.load_state_dict(ck["model"], strict=False)
    model = model.to(device).eval()
    tok = Tokenizer.from_file(args.tokenizer)
    scorer = BERTScorer(lang="en", rescale_with_baseline=False)
    print(f"arch {arch} | train BERTScore {ck.get('bert_f1', float('nan')):.4f}")

    dev = load_split("dev")[:args.n]
    refs = [d["summary"] for d in dev]

    chosen, best = None, (-1.0, None)
    print(f"\n  {'config':<22} {'len':>6} {'rep4':>6} {'BERTScore':>10}")
    for cfg in CONFIGS:
        if args.config and cfg[0] != args.config:
            continue
        t0 = time.time()
        cands, ln = run(model, dev, tok, device, ctx, arch, cfg, args.batch)
        F = scorer.score(cands, refs)[2].mean().item()
        rr = float(np.mean([rep_rate(c) for c in cands]))
        print(f"  {cfg[0]:<22} {ln:>6.0f} {rr:>6.3f} {F:>10.4f}"
              f"   ({time.time()-t0:.0f}s)")
        if F > best[0]:
            best, chosen = (F, cfg), cfg
    print(f"\n  best: {chosen[0]} at {best[0]:.4f}  "
          f"(lead-512 baseline 0.8300)")

    if not args.sub_name:
        return

    test = load_split("test")
    cands, ln = run(model, test, tok, device, ctx, arch, chosen, args.batch)
    preds = [{"id": d["id"], "summary": c} for d, c in zip(test, cands)]
    assert len(preds) == len(test)

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    jp = os.path.join(sub_dir, "prediction.json")
    with open(jp, "w") as f:
        json.dump(preds, f)
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          6 (summarization)\n")
        f.write(f"checkpoint     {args.checkpoint}\n")
        f.write(f"arch {arch} | decoding {chosen[0]}\n")
        f.write(f"dev BERTScore  {best[0]:.4f} (lead-512 baseline 0.8300)\n")
        f.write(f"mean gen len   {ln:.0f} tokens\n")
        if args.note:
            f.write(f"note           {args.note}\n")
    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")
    print(f"\n{sub_dir}/ written — {len(preds)} summaries, mean {ln:.0f} tokens")


if __name__ == "__main__":
    main()