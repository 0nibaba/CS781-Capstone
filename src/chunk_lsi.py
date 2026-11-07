import os, json, argparse, zipfile, datetime

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from lsi_data import load_split, TOKENIZER, N_LABELS, PAD_ID
from lsi_model import StatuteClassifier
from lsi_metrics import scores, best_global, best_per_label
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chunks_of(ids, max_len, stride, max_chunks):
    if len(ids) <= max_len:
        return [ids]
    out = []
    for s in range(0, len(ids), stride):
        c = ids[s:s + max_len]
        if len(c) < 64 and out:
            break
        out.append(c)
        if len(out) >= max_chunks or s + max_len >= len(ids):
            break
    return out


@torch.no_grad()
def doc_probs(model, docs, tok, device, ctx, max_len, stride, max_chunks,
              batch_size):
    flat, owner = [], []
    for i, d in enumerate(docs):
        for c in chunks_of(tok.encode(d["text"]).ids, max_len, stride,
                           max_chunks):
            flat.append(c)
            owner.append(i)
    print(f"  {len(docs)} docs -> {len(flat)} chunks "
          f"({len(flat)/len(docs):.2f}/doc)")

    P = np.zeros((len(flat), N_LABELS), dtype=np.float32)
    order = np.argsort([len(c) for c in flat])      # bucket by length
    for b in range(0, len(order), batch_size):
        sel = order[b:b + batch_size]
        n = max(len(flat[j]) for j in sel)
        ids = torch.full((len(sel), n), PAD_ID, dtype=torch.long)
        msk = torch.zeros(len(sel), n, dtype=torch.long)
        for r, j in enumerate(sel):
            L = len(flat[j])
            ids[r, :L] = torch.tensor(flat[j], dtype=torch.long)
            msk[r, :L] = 1
        with ctx:
            logits, _ = model(ids.to(device), msk.to(device))
        P[sel] = torch.sigmoid(logits.float()).cpu().numpy()

    owner = np.array(owner)
    out = np.zeros((len(docs), 3, N_LABELS), dtype=np.float32)
    for i in range(len(docs)):
        p = P[owner == i]
        out[i, 0] = p.max(0)
        out[i, 1] = p.mean(0)
        out[i, 2] = 1.0 - np.prod(1.0 - p, axis=0)   # noisy-or
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--sub_name", default="")
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--max_chunks", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--note", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    max_len = tr.get("max_len", 1024)
    pooling = tr.get("pooling", "mean")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = StatuteClassifier(base, arch, d_model, dropout=0.0,
                              pooling=pooling)
    if "keep_every" in ck:
        ke = ck["keep_every"]
        blocks = model.base.transformer.h
        model.base.transformer.h = nn.ModuleList(
            [blocks[i] for i in range(0, len(blocks), ke)])
        model.base.config.n_layer = len(model.base.transformer.h)
    if tr.get("finetune") == "lorma" or any(".c_attn.A." in k
                                            for k in ck["model"]):
        from lorma import apply_lorma
        apply_lorma(model.base, r=tr.get("lorma_r", 8),
                    alpha=tr.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
    model.load_state_dict(ck["model"], strict=False)
    model = model.to(device).eval()
    print(f"arch {ck['arch']} | trained single-window "
          f"(dev {ck.get('macro_f1', float('nan')):.4f})")

    tok = Tokenizer.from_file(args.tokenizer)
    names = ["max", "mean", "noisy-or"]

    dev = load_split("dev")
    print("\ndev:")
    D = doc_probs(model, dev, tok, device, ctx, max_len, args.stride,
                  args.max_chunks, args.batch_size)
    gold = np.zeros((len(dev), N_LABELS), dtype=int)
    for i, d in enumerate(dev):
        for v in d["labels"]:
            gold[i, v] = 1

    best = (-1.0, None, None)
    print(f"\n  {'aggregator':<12} {'@0.5':>8} {'global':>8} {'per-label':>10}")
    for a, nm in enumerate(names):
        m05 = scores(D[:, a], gold, 0.5)[0]
        mg, _ = best_global(D[:, a], gold)
        mpl, thr = best_per_label(D[:, a], gold)
        print(f"  {nm:<12} {m05:>8.4f} {mg:>8.4f} {mpl:>10.4f}")
        if mpl > best[0]:
            best = (mpl, a, thr)
    mpl, a, thr = best
    print(f"\n  best: {names[a]} at {mpl:.4f} "
          f"(single-window baseline {ck.get('macro_f1', float('nan')):.4f})")

    if not args.sub_name:
        return

    test = load_split("test")
    print("\ntest:")
    T = doc_probs(model, test, tok, device, ctx, max_len, args.stride,
                  args.max_chunks, args.batch_size)
    hit = T[:, a] >= thr
    preds, forced = [], 0
    for i, d in enumerate(test):
        ids_ = sorted(int(j) for j in np.where(hit[i])[0])
        if not ids_:
            ids_ = [int(T[i, a].argmax())]
            forced += 1
        preds.append({"id": d["id"], "pred": ids_})
    n_per = np.array([len(x["pred"]) for x in preds])
    print(f"  labels/doc mean {n_per.mean():.2f} | empty-forced {forced}")

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    jp = os.path.join(sub_dir, "prediction.json")
    with open(jp, "w") as f:
        json.dump(preds, f)
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          5 (LSI)\n")
        f.write(f"method         chunked inference, {names[a]} aggregation\n")
        f.write(f"checkpoint     {args.checkpoint}\n")
        f.write(f"stride {args.stride} max_chunks {args.max_chunks} "
                f"max_len {max_len}\n")
        f.write(f"dev macro-F1   {mpl:.4f} "
                f"(single-window {ck.get('macro_f1', float('nan')):.4f})\n")
        if args.note:
            f.write(f"note           {args.note}\n")
        f.write(f"\n{len(preds)} docs, mean {n_per.mean():.2f} labels/doc\n")
    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")
    print(f"\n{sub_dir}/ written")


if __name__ == "__main__":
    main()