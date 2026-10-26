"""Probability-averaging ensemble over LSI models from different Phase 1 archs.

Phase 3's vote ensemble gained +0.006 and I attributed that to members sharing
one pretrained checkpoint and 80 training documents, so their errors were
correlated. This tests that claim directly: four DIFFERENT Phase 1
architectures, four adaptation techniques, 42,750 documents.

Per-label thresholds are re-tuned on the averaged probabilities, since averaging
changes the score scale.
"""
import os, json, argparse, zipfile, datetime

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from lsi_data import LSIDocs, load_split, batches, TOKENIZER, N_LABELS
from lsi_model import StatuteClassifier
from lsi_metrics import scores, best_global, best_per_label
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(ck_path, base_path, device):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    base, arch, d_model = load_base(base_path, device)
    m = StatuteClassifier(base, arch, d_model, dropout=0.0,
                          pooling=tr.get("pooling", "mean"))
    if "keep_every" in ck:
        ke = ck["keep_every"]
        b = m.base.transformer.h
        m.base.transformer.h = nn.ModuleList(
            [b[i] for i in range(0, len(b), ke)])
        m.base.config.n_layer = len(m.base.transformer.h)
    if tr.get("finetune") == "lorma" or any(".c_attn.A." in k
                                            for k in ck["model"]):
        from lorma import apply_lorma
        apply_lorma(m.base, r=tr.get("lorma_r", 8),
                    alpha=tr.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected
    return m.to(device).eval(), ck, tr


@torch.no_grad()
def probs_for(model, docs, tok, mode, max_len, device, ctx, batch_size):
    ds = LSIDocs(docs, tok, mode=mode, max_len=max_len)
    P = []
    for chunk, (ids, msk, y) in batches(ds, batch_size, device,
                                        shuffle=False):
        with ctx:
            logits, _ = model(ids, msk)
        P.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(P).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="ckpt_path:base_path pairs")
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--sub_name", default="")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--note", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    tok = Tokenizer.from_file(args.tokenizer)

    dev = load_split("dev")
    gold = np.zeros((len(dev), N_LABELS), dtype=int)
    for i, d in enumerate(dev):
        for v in d["labels"]:
            gold[i, v] = 1
    test = load_split("test")

    dev_P, test_P, names = [], [], []
    for spec in args.models:
        ck_path, base_path = spec.split(":")
        model, ck, tr = build(ck_path, base_path, device)
        name = os.path.basename(os.path.dirname(ck_path))
        mode, max_len = tr.get("mode", "head"), tr.get("max_len", 1024)
        dp = probs_for(model, dev, tok, mode, max_len, device, ctx,
                       args.batch_size)
        mpl, _ = best_per_label(dp, gold)
        print(f"  {name:<14} {ck['arch']:<10} dev {mpl:.4f}")
        dev_P.append(dp)
        names.append(name)
        if args.sub_name:
            test_P.append(probs_for(model, test, tok, mode, max_len, device,
                                    ctx, args.batch_size))
        del model
        torch.cuda.empty_cache()

    D = np.mean(dev_P, axis=0)
    m05 = scores(D, gold, 0.5)[0]
    mg, tg = best_global(D, gold)
    mpl, thr = best_per_label(D, gold)
    best_single = max(best_per_label(x, gold)[0] for x in dev_P)
    print(f"\n  ENSEMBLE of {len(dev_P)}: @0.5 {m05:.4f} | "
          f"global {mg:.4f}@{tg:.2f} | per-label {mpl:.4f}")
    print(f"  best single member: {best_single:.4f}  "
          f"(gain {mpl - best_single:+.4f})")

    # how decorrelated are the members, really?
    print("\n  pairwise correlation of member probabilities:")
    for i in range(len(dev_P)):
        for j in range(i + 1, len(dev_P)):
            c = np.corrcoef(dev_P[i].ravel(), dev_P[j].ravel())[0, 1]
            print(f"    {names[i]:<14} {names[j]:<14} {c:.3f}")

    if not args.sub_name:
        return

    T = np.mean(test_P, axis=0)
    hit = T >= thr
    preds, forced = [], 0
    for i, d in enumerate(test):
        ids_ = sorted(int(j) for j in np.where(hit[i])[0])
        if not ids_:
            ids_ = [int(T[i].argmax())]
            forced += 1
        preds.append({"id": d["id"], "pred": ids_})
    n_per = np.array([len(x["pred"]) for x in preds])
    print(f"\n  labels/doc mean {n_per.mean():.2f} | empty-forced {forced}")

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    jp = os.path.join(sub_dir, "prediction.json")
    with open(jp, "w") as f:
        json.dump(preds, f)
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          5 (LSI)\n")
        f.write(f"method         probability-averaging ensemble, "
                f"per-label thresholds\n")
        for s in args.models:
            f.write(f"  member       {s}\n")
        f.write(f"dev macro-F1   {mpl:.4f} "
                f"(best single {best_single:.4f})\n")
        if args.note:
            f.write(f"note           {args.note}\n")
        f.write(f"\n{len(preds)} docs, mean {n_per.mean():.2f} labels/doc\n")
    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")
    print(f"\n{sub_dir}/ written")


if __name__ == "__main__":
    main()