"""Phase 3 submission: {doc_id: [label, ...]} with one label per sentence.

Writes submissions/<sub_name>/{prediction.json, config.txt, submission.zip},
matching the Phase 1 and Phase 2 layout.
"""
import os, json, glob, argparse, zipfile, datetime
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from rr_data import RRBlocks, batches, TOKENIZER, ID2LABEL, LABELS
from rr_model import SentenceClassifier
from ner_model import load_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, "test_data/Phase_3_RR/Test")


def load_test():
    """Test files are {doc_id: [sentence, ...]} -- train is a list of dicts."""
    docs = []
    for path in sorted(glob.glob(os.path.join(TEST_DIR, "*.json"))):
        dom = "CL" if "CL" in os.path.basename(path).upper() else "IT"
        with open(path) as f:
            d = json.load(f)
        assert isinstance(d, dict), f"{path}: expected a dict"
        for did, sents in d.items():
            docs.append({"id": did, "domain": dom, "text": sents,
                         # RRBlocks wants labels; dummies are never read back
                         "labels": [LABELS[0]] * len(sents)})
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub_name", required=True, help="e.g. p3_sub1")
    p.add_argument("--rr_checkpoint", required=True)
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--json_name", default="prediction.json",
                   help="filename inside the zip; must match the grader")
    p.add_argument("--note", default="", help="free text for config.txt")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--ctx_frac", type=float, default=0.34)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ck = torch.load(args.rr_checkpoint, map_location="cpu")
    has_crf = any(k.startswith("crf.") for k in ck["model"])
    dev_f1 = ck.get("macro_f1", float("nan"))
    print(f"arch {ck['arch']} | crf {has_crf} | dev macro-F1 "
          f"{dev_f1:.4f} @ epoch {ck.get('epoch')}")

    base, arch, d_model = load_base(args.base_checkpoint, device)
    model = SentenceClassifier(base, arch, d_model, dropout=0.0,
                               use_crf=has_crf)
    if "keep_every" in ck:
        ke = ck["keep_every"]
        blocks = model.base.transformer.h
        model.base.transformer.h = nn.ModuleList(
            [blocks[i] for i in range(0, len(blocks), ke)])
        model.base.config.n_layer = len(model.base.transformer.h)
        print(f"distilled student: kept every {ke} -> "
              f"{model.base.config.n_layer} layers")
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
    assert not missing, f"missing keys: {list(missing)[:5]}"
    model = model.to(device).eval()

    docs = load_test()
    tok = Tokenizer.from_file(args.tokenizer)
    # block geometry must match training exactly; take it from the checkpoint
    tr_args = ck.get("args", {})
    n_ctx = tr_args.get("context", args.context)
    frac = tr_args.get("ctx_frac", args.ctx_frac)
    mx = tr_args.get("max_len", 1024)
    print(f"block geometry from checkpoint: context={n_ctx} ctx_frac={frac} "
          f"max_len={mx}")
    ds = RRBlocks(docs, tok, max_len=mx, context=n_ctx, ctx_frac=frac)

    out = {d["id"]: [None] * len(d["text"]) for d in docs}
    with torch.no_grad():
        for chunk, (ids, msk, pos, pmask, lab) in batches(
                ds, args.batch_size, device, shuffle=False):
            with ctx:
                logits, _ = model(ids, msk, pos, pmask)
            paths = model.decode(logits, pmask)
            for b, it in enumerate(chunk):
                for j, si in enumerate(range(it["pred_from"], it["pred_to"])):
                    out[it["doc_id"]][si] = ID2LABEL[paths[b][j]]

    # the grader requires exactly one label per sentence, in order
    for d in docs:
        v = out[d["id"]]
        assert len(v) == len(d["text"]), \
            f"{d['id']}: {len(v)} labels for {len(d['text'])} sentences"
        assert all(x is not None for x in v), \
            f"{d['id']}: {sum(x is None for x in v)} sentences unpredicted"
        assert all(x in LABELS for x in v), f"{d['id']}: bad label"

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    json_path = os.path.join(sub_dir, args.json_name)
    with open(json_path, "w") as f:
        json.dump(out, f)

    n_sents = sum(len(v) for v in out.values())
    dist = {}
    for v in out.values():
        for x in v:
            dist[x] = dist.get(x, 0) + 1

    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"phase          3 (Rhetorical Role)\n\n")
        f.write(f"rr_checkpoint  {args.rr_checkpoint}\n")
        f.write(f"base_checkpoint {args.base_checkpoint}\n")
        f.write(f"arch           {ck['arch']}\n")
        f.write(f"block geometry context={ctx} ctx_frac={frac} max_len={mx}\n")
        f.write(f"crf            {has_crf}\n")
        f.write(f"dev macro-F1   {dev_f1:.4f} @ epoch {ck.get('epoch')}\n")
        if args.note:
            f.write(f"note           {args.note}\n")
        f.write(f"\ntrain args:\n")
        for k, v in sorted(ck.get("args", {}).items()):
            f.write(f"  {k:18s} {v}\n")
        f.write(f"\n{len(out)} docs / {n_sents} sentences\n")
        f.write("predicted label distribution:\n")
        for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
            f.write(f"  {k:26s} {v:6d}  {100*v/n_sents:5.2f}%\n")

    zip_path = os.path.join(sub_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname=args.json_name)

    print(f"\n{sub_dir}/")
    print(f"  {args.json_name}   {len(out)} docs / {n_sents} sentences")
    print(f"  config.txt")
    print(f"  submission.zip")
    print("\npredicted label distribution:")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1])[:5]:
        f1 = 100 * v / n_sents
        print(f"  {k:26s} {v:6d}  {f1:5.2f}%")


if __name__ == "__main__":
    main()