import os
import json
import argparse

import numpy as np
import torch
from tokenizers import Tokenizer

from ner_model import TokenClassifier, load_base
from ner_data import NERWindows, load_fold, batches, TOKENIZER
from windowing import merge_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="checkpoints/ner/<run>_<fold> containing ckpt_best.pt")
    ap.add_argument("--fold", required=True, choices=["fold1", "fold2", "fold3"])
    ap.add_argument("--base_checkpoint", required=True,
                    help="the Phase 1 checkpoint the head was trained on")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_position", action="store_true",
                    help="must match how the checkpoint was trained")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    tok = Tokenizer.from_file(TOKENIZER)
    docs, gold = load_fold(args.fold)
    ds = NERWindows(docs, gold, tok)
    print(f"{args.fold}: {len(docs)} documents -> {len(ds)} windows")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = TokenClassifier(base, arch, d_model, use_position=args.use_position).to(device)
    ck = torch.load(os.path.join(args.run_dir, "ckpt_best.pt"),
                    map_location="cpu", weights_only=False)
    trained_with_pos = ck["args"].get("use_position", False)
    if trained_with_pos != args.use_position:
        raise SystemExit(
            f"checkpoint was trained with use_position={trained_with_pos}, "
            f"but --use_position was {'given' if args.use_position else 'omitted'}")
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded epoch {ck['epoch']}, macro-F1 {ck['macro_f1']:.4f}")

    per_doc = {}
    for chunk, (ids, msk, lab) in batches(ds, args.batch_size, device,
                                          shuffle=False):
        with ctx:
            logits, _ = model(ids, attn_mask=msk, doc_pos = pos)
        probs = torch.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(-1)
        pred, conf = pred.cpu(), conf.cpu()

        for i, it in enumerate(chunk):
            n = len(it["input_ids"])
            per_doc.setdefault(it["doc_id"], []).append({
                "tags": pred[i, :n].tolist(),
                "conf": conf[i, :n].tolist(),
                "tok_start": it["tok_start"],
                "keep_from": it["keep_from"],
                "keep_to": it["keep_to"],
                "n_tokens": it["n_tokens"],
            })

    out = {}
    for did, wins in per_doc.items():
        n_tok = wins[0]["n_tokens"]
        out[did] = {
            "tags": merge_windows(wins, n_tok, "tags"),
            "conf": merge_windows_f(wins, n_tok),
        }

    path = os.path.join(args.run_dir, f"tags_{args.fold}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"saved {len(out)} documents -> {path}")
    print(f"  {sum(len(v['tags']) for v in out.values()):,} tokens")


def merge_windows_f(windows, n_tokens):
    full = [0.0] * n_tokens
    for w in windows:
        s = w["tok_start"]
        for i in range(w["keep_from"], w["keep_to"]):
            full[s + i] = w["conf"][i]
    return full


if __name__ == "__main__":
    main()