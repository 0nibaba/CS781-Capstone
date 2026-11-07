import os
import json
import zipfile
import argparse

import torch
from tokenizers import Tokenizer

from ner_model import TokenClassifier, load_base
from ner_data import NERWindows, load_fold, batches, TOKENIZER
from windowing import merge_windows
from decode_variants import decode
from bio_tagging import LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="names the output directory")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lenient", action="store_true")
    ap.add_argument("--bridge", type=int, default=3)
    ap.add_argument("--min_chars", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    tok = Tokenizer.from_file(TOKENIZER)
    docs, _ = load_fold("fold3")
    ds = NERWindows(docs, None, tok)
    print(f"fold3: {len(docs)} documents -> {len(ds)} windows")

    ckpt = os.path.join(args.run_dir, "ckpt_final.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(args.run_dir, "ckpt_best.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    tr_args = ck["args"]

    base, arch, d_model = load_base(args.base_checkpoint, device)
    has_crf = any(k.startswith("crf.") for k in ck["model"])
    model = TokenClassifier(base, arch, d_model,
                            use_position=tr_args.get("use_position", False),
                            use_crf=has_crf).to(device)
    if tr_args.get("lorma"):
        from lorma import apply_lorma
        apply_lorma(model, r=tr_args.get("lorma_r", 8),
                    alpha=tr_args.get("lorma_alpha", 16.0), verbose=False)
        model.to(device)

    if tr_args.get("keep_every") and tr_args.get("teacher_dir"):
        from distill_ner import make_student
        model = make_student(model, tr_args["keep_every"]).to(device)
        print(f"  truncated to {len(model.base.transformer.h)} layers")

    model.load_state_dict(ck["model"])
        
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {os.path.basename(ckpt)}: arch {arch}, "
          f"crf={tr_args.get('use_crf')}, pos={tr_args.get('use_position')}")

    per_doc = {}
    for chunk, (ids, msk, lab, pos) in batches(ds, args.batch_size, device,
                                               shuffle=False):
        with ctx:
            logits, _ = model(ids, attn_mask=msk, doc_pos=pos)
        if model.use_crf:
            paths = model.crf.viterbi(logits.float(), msk)
        else:
            argmax = logits.argmax(-1).cpu()
        for i, it in enumerate(chunk):
            n = len(it["input_ids"])
            tags = paths[i][:n] if model.use_crf else argmax[i, :n].tolist()
            per_doc.setdefault(it["doc_id"], []).append({
                "tags": tags, "tok_start": it["tok_start"],
                "keep_from": it["keep_from"], "keep_to": it["keep_to"],
                "n_tokens": it["n_tokens"],
            })

    preds = {}
    for did, text in docs.items():
        wins = per_doc[did]
        merged = merge_windows(wins, wins[0]["n_tokens"])
        preds[did] = decode(merged, ds.offsets[did], text,
                            lenient=args.lenient, bridge=args.bridge,
                            min_chars=args.min_chars)

    print("\nvalidating...")
    assert set(preds) == set(docs), "document ids do not match the test set"
    n = sum(len(v) for v in preds.values())
    bad = 0
    for did, spans in preds.items():
        L = len(docs[did])
        for s, e, lab in spans:
            if not (0 <= s < e <= L) or lab not in LABELS:
                bad += 1
    assert bad == 0, f"{bad} malformed spans"
    print(f"  {len(preds)} documents, {n:,} entities "
          f"({n/len(preds):.0f} per document)")
    print("  training data had ~105 per document")

    from collections import Counter
    c = Counter(l for v in preds.values() for _, _, l in v)
    print("\n  label distribution:")
    for lab in LABELS:
        print(f"    {lab:<12} {c.get(lab, 0):>6}"
              + ("   <-- none predicted" if not c.get(lab) else ""))

    out_dir = os.path.join(ROOT, "submissions", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "prediction.json")
    zip_path = os.path.join(out_dir, "submission.zip")

    with open(pred_path, "w") as f:
        json.dump(preds, f)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], "bad archive layout"

    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        f.write(f"checkpoint: {ckpt}\nbase: {args.base_checkpoint}\n"
                f"arch: {arch}\ncrf: {tr_args.get('use_crf')}\n"
                f"position: {tr_args.get('use_position')}\n"
                f"decode: lenient={args.lenient} bridge={args.bridge} "
                f"min_chars={args.min_chars}\nentities: {n}\n")

    print(f"\n-> {zip_path}")


if __name__ == "__main__":
    main()