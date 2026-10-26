"""Sweep logit-adjustment tau on dev.

Macro-F1 under a long-tailed prior is maximised by a decision rule that is not
argmax of the posterior. Subtracting tau*log p(y) trades majority-class precision
for minority recall, which is exactly the trade macro-F1 rewards.

CAVEAT: dev has no Dissent or PrecedentOverruled, so this can only verify that
tau does not DAMAGE the 11 visible labels. The gain on the other two is unmeasurable
here and must be bought with a leaderboard submission.
"""
import os, argparse
from collections import Counter
import torch
from tokenizers import Tokenizer

from rr_data import (RRBlocks, load_split, batches, TOKENIZER,
                     ID2LABEL, LABELS, N_LABELS)
from rr_metrics import macro_scores
from rr_model import SentenceClassifier
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_log_prior(device):
    c = Counter(l for d in load_split("train", ("CL", "IT")) for l in d["labels"])
    n = sum(c.values())
    p = torch.tensor([max(c.get(ID2LABEL[i], 0), 1) / n for i in range(N_LABELS)])
    return p.log().to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rr_checkpoint", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--taus", default="0,0.1,0.2,0.3,0.4,0.5,0.75,1.0")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ck = torch.load(args.rr_checkpoint, map_location="cpu")
    has_crf = any(k.startswith("crf.") for k in ck["model"])
    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = SentenceClassifier(base, arch, d_model, dropout=0.0, use_crf=has_crf)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(device).eval()

    docs = load_split("dev", ("CL", "IT"))
    tok = Tokenizer.from_file(args.tokenizer)
    ds = RRBlocks(docs, tok)
    gold = {d["id"]: d["labels"] for d in docs}
    gold_labels = {l for v in gold.values() for l in v}
    print(f"dev {len(docs)} docs / {sum(len(v) for v in gold.values())} sents, "
          f"{len(gold_labels)}/{N_LABELS} labels present\n")

    # cache logits once; tau only changes decoding
    cache = []
    with torch.no_grad():
        for chunk, (ids, msk, pos, pmask, lab) in batches(
                ds, args.batch_size, device, shuffle=False):
            with ctx:
                logits, _ = model(ids, msk, pos, pmask)
            cache.append((chunk, logits.float().cpu(), pmask.cpu()))

    logp = train_log_prior("cpu")

    print(f"{'tau':>5} {'macro(11)':>10} {'acc':>7} {'ratio%':>7}  rare predictions")
    for tau in [float(t) for t in args.taus.split(",")]:
        out = {d["id"]: [None] * len(d["text"]) for d in docs}
        for chunk, logits, pmask in cache:
            adj = logits - tau * logp
            paths = model.decode(adj.to(device), pmask.to(device))
            for b, it in enumerate(chunk):
                for j, si in enumerate(range(it["pred_from"], it["pred_to"])):
                    out[it["doc_id"]][si] = ID2LABEL[paths[b][j]]

        mf, mp, mr, acc, per = macro_scores(out, gold)
        # restrict the average to labels dev can actually score, so that
        # predicting an absent label does not fake a drop
        def _f1(v):
            return v["f1"] if isinstance(v, dict) else v[2]
        vis = [_f1(per[l]) for l in gold_labels if l in per]
        macro11 = sum(vis) / len(vis)

        cnt = Counter(x for v in out.values() for x in v)
        n = sum(cnt.values())
        rare = " ".join(f"{l}:{cnt.get(l,0)}" for l in
                        ("Dissent", "PrecedentOverruled", "Issue",
                         "PrecedentNotReliedUpon"))
        print(f"{tau:5.2f} {macro11:10.4f} {acc:7.4f} "
              f"{100*cnt.get('RatioOfTheDecision',0)/n:6.2f}%  {rare}")


if __name__ == "__main__":
    main()