"""Phase 5 submission: [{"id": ..., "pred": [int, ...]}, ...]"""
import os, json, argparse, zipfile, datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from lsi_data import LSIDocs, load_split, batches, TOKENIZER, N_LABELS
from lsi_model import StatuteClassifier
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--json_name", default="prediction.json")
    p.add_argument("--threshold", default="per_label",
                   choices=["per_label", "global", "half"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--note", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    mode = tr.get("mode", "head")
    pooling = tr.get("pooling", "mean")
    max_len = tr.get("max_len", 1024)
    print(f"arch {ck['arch']} | mode {mode} | pooling {pooling} | "
          f"dev macro-F1 {ck.get('macro_f1', float('nan')):.4f}")

    if args.threshold == "per_label":
        thr = np.array(ck["thresholds"])
    elif args.threshold == "global":
        thr = float(ck["thr_global"])
    else:
        thr = 0.5
    print(f"threshold rule: {args.threshold} "
          f"({'vector' if isinstance(thr, np.ndarray) else thr})")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = StatuteClassifier(base, arch, d_model, dropout=0.0,
                              pooling=pooling)
    if "keep_every" in ck:
        ke = ck["keep_every"]
        blocks = model.base.transformer.h
        model.base.transformer.h = nn.ModuleList(
            [blocks[i] for i in range(0, len(blocks), ke)])
        model.base.config.n_layer = len(model.base.transformer.h)
        print(f"distilled student: {model.base.config.n_layer} layers")
    if tr.get("finetune") == "lorma" or any(".c_attn.A." in k
                                            for k in ck["model"]):
        from lorma import apply_lorma
        apply_lorma(model.base, r=tr.get("lorma_r", 8),
                    alpha=tr.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
        print("LoRMA adapters installed")
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (list(missing)[:5],
                                            list(unexpected)[:5])
    model = model.to(device).eval()

    docs = load_split("test")
    tok = Tokenizer.from_file(args.tokenizer)
    ds = LSIDocs(docs, tok, mode=mode, max_len=max_len)
    print(f"test {len(ds)} documents")

    P = []
    with torch.no_grad():
        for chunk, (ids, msk, y) in batches(ds, args.batch_size, device,
                                            shuffle=False):
            with ctx:
                logits, _ = model(ids, msk)
            P.append(torch.sigmoid(logits.float()).cpu())
    prob = torch.cat(P).numpy()
    assert prob.shape == (len(docs), N_LABELS)

    hit = prob >= thr
    preds = []
    empty = 0
    for i, d in enumerate(docs):
        ids_ = sorted(int(j) for j in np.where(hit[i])[0])
        if not ids_:                       # never submit an empty set
            ids_ = [int(prob[i].argmax())]
            empty += 1
        assert len(ids_) == len(set(ids_))
        preds.append({"id": d["id"], "pred": ids_})

    n_per = np.array([len(x["pred"]) for x in preds])
    print(f"labels/doc: mean {n_per.mean():.2f} median {np.median(n_per):.0f} "
          f"max {n_per.max()} | empty-forced {empty}")

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    json_path = os.path.join(sub_dir, args.json_name)
    with open(json_path, "w") as f:
        json.dump(preds, f)

    c = Counter(v for x in preds for v in x["pred"])
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          5 (LSI)\n")
        f.write(f"checkpoint     {args.checkpoint}\n")
        f.write(f"base           {args.base_checkpoint}\n")
        f.write(f"arch {ck['arch']} mode {mode} pooling {pooling} "
                f"max_len {max_len}\n")
        f.write(f"threshold rule {args.threshold}\n")
        f.write(f"dev macro-F1   {ck.get('macro_f1', float('nan')):.4f} "
                f"(@0.5 {ck.get('macro_05', float('nan')):.4f}, "
                f"global {ck.get('macro_global', float('nan')):.4f})\n")
        if args.note:
            f.write(f"note           {args.note}\n")
        for k, v in sorted(tr.items()):
            f.write(f"  {k:18s} {v}\n")
        f.write(f"\n{len(preds)} docs, {int(n_per.sum())} label assignments, "
                f"mean {n_per.mean():.2f}/doc, {len(c)} distinct statutes\n")

    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname=args.json_name)
    print(f"\n{sub_dir}/ written")


if __name__ == "__main__":
    main()