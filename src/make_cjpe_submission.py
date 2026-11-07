import os, json, argparse, zipfile, datetime
from collections import Counter

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from cjpe_data import CJPEDocs, load_split, batches, TOKENIZER
from cjpe_model import DocClassifier
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# training prior, from profile_phase4: 22067 / 15320 of 37387
TRAIN_PRIOR = torch.tensor([22067 / 37387, 15320 / 37387])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub_name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--json_name", default="prediction.json")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--note", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tr = ck.get("args", {})
    mode = tr.get("mode", "tail")
    pooling = tr.get("pooling", "last")
    max_len = tr.get("max_len", 1024)
    decision = ck.get("decision", "raw")
    print(f"arch {ck['arch']} | mode {mode} | pooling {pooling} | "
          f"max_len {max_len} | decision {decision} | "
          f"dev {ck.get('macro_f1', float('nan')):.4f}")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = DocClassifier(base, arch, d_model, dropout=0.0, pooling=pooling)
    if "keep_every" in ck:
        ke = ck["keep_every"]
        blocks = model.base.transformer.h
        model.base.transformer.h = nn.ModuleList(
            [blocks[i] for i in range(0, len(blocks), ke)])
        model.base.config.n_layer = len(model.base.transformer.h)
        print(f"distilled student: kept every {ke} -> "
              f"{model.base.config.n_layer} layers")
    if tr.get("finetune") == "lorma" or any(".c_attn.A." in k
                                        for k in ck["model"]):
        from lorma import apply_lorma
        apply_lorma(model.base, r=tr.get("lorma_r", 8),
                    alpha=tr.get("lorma_alpha", 16.0),
                    targets=("q", "v"), verbose=False)
        print("LoRMA adapters installed before loading state dict")
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (list(missing)[:5],
                                            list(unexpected)[:5])
    model = model.to(device).eval()

    docs = load_split("test")
    assert all("label" not in d for d in docs),
    tok = Tokenizer.from_file(args.tokenizer)
    ds = CJPEDocs(docs, tok, mode=mode, max_len=max_len)
    print(f"test {len(ds)} cases")

    log_prior = TRAIN_PRIOR.log()
    out = {}
    with torch.no_grad():
        for chunk, (ids, msk, lab) in batches(ds, args.batch_size, device,
                                              shuffle=False):
            with ctx:
                logits, _ = model(ids, msk)
            L = logits.float().cpu()
            if decision == "adj":
                L = L - log_prior
            for it, pr in zip(chunk, L.argmax(-1).tolist()):
                out[it["id"]] = int(pr)

    assert len(out) == len(docs), f"{len(out)} preds for {len(docs)} cases"
    preds = [{"id": d["id"], "pred": out[d["id"]]} for d in docs]
    assert all(x["pred"] in (0, 1) for x in preds)

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    json_path = os.path.join(sub_dir, args.json_name)
    with open(json_path, "w") as f:
        json.dump(preds, f)

    c = Counter(x["pred"] for x in preds)
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          4 (CJPE)\n")
        f.write(f"checkpoint     {args.checkpoint}\n")
        f.write(f"base           {args.base_checkpoint}\n")
        f.write(f"arch {ck['arch']} mode {mode} pooling {pooling} "
                f"max_len {max_len} decision {decision}\n")
        f.write(f"dev macro-F1   {ck.get('macro_f1', float('nan')):.4f}\n")
        if args.note:
            f.write(f"note           {args.note}\n")
        for k, v in sorted(tr.items()):
            f.write(f"  {k:18s} {v}\n")
        f.write(f"\n{len(preds)} predictions: "
                f"0 -> {c[0]} ({100*c[0]/len(preds):.1f}%), "
                f"1 -> {c[1]} ({100*c[1]/len(preds):.1f}%)\n")

    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname=args.json_name)

    print(f"\n{sub_dir}/ written")
    print(f"  0 -> {c[0]} ({100*c[0]/len(preds):.1f}%), "
          f"1 -> {c[1]} ({100*c[1]/len(preds):.1f}%)")


if __name__ == "__main__":
    main()