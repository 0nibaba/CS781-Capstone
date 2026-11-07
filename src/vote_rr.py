import os, json, argparse, zipfile, datetime
from collections import Counter
import torch
from tokenizers import Tokenizer

from rr_data import RRBlocks, batches, TOKENIZER, ID2LABEL, LABELS
from rr_model import SentenceClassifier
from ner_model import load_base
from make_rr_submission import load_test, ROOT


def predict_one(rr_ck, base_ck, ds, docs, device, ctx, batch_size):
    ck = torch.load(rr_ck, map_location="cpu")
    has_crf = any(k.startswith("crf.") for k in ck["model"])
    base, arch, d_model = load_base(base_ck, device)
    model = SentenceClassifier(base, arch, d_model, dropout=0.0, use_crf=has_crf)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(device).eval()
    print(f"  {os.path.basename(os.path.dirname(rr_ck)):22s} "
          f"crf={has_crf} dev={ck.get('macro_f1', float('nan')):.4f}")

    out = {d["id"]: [None] * len(d["text"]) for d in docs}
    with torch.no_grad():
        for chunk, (ids, msk, pos, pmask, lab) in batches(
                ds, batch_size, device, shuffle=False):
            with ctx:
                logits, _ = model(ids, msk, pos, pmask)
            paths = model.decode(logits, pmask)
            for b, it in enumerate(chunk):
                for j, si in enumerate(range(it["pred_from"], it["pred_to"])):
                    out[it["doc_id"]][si] = ID2LABEL[paths[b][j]]
    del model, base
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub_name", required=True)
    p.add_argument("--rr_checkpoints", nargs="+", required=True,
                   help="best-first; ties break toward the first")
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--json_name", default="prediction.json")
    p.add_argument("--note", default="")
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    docs = load_test()
    tok = Tokenizer.from_file(args.tokenizer)
    ds = RRBlocks(docs, tok)
    print(f"test {len(docs)} docs -> {len(ds)} blocks")
    print(f"ensemble of {len(args.rr_checkpoints)}:")

    members = [predict_one(c, args.base_checkpoint, ds, docs, device, ctx,
                           args.batch_size) for c in args.rr_checkpoints]

    out, changed = {}, 0
    for d in docs:
        did, seq = d["id"], []
        for si in range(len(d["text"])):
            votes = [m[did][si] for m in members]
            top = Counter(votes).most_common()
            best = max(c for _, c in top)
            tied = {l for l, c in top if c == best}
            pick = next(v for v in votes if v in tied)
            seq.append(pick)
            if pick != votes[0]:
                changed += 1
        out[did] = seq

    n = sum(len(v) for v in out.values())
    print(f"\nvote changed {changed}/{n} ({100*changed/n:.2f}%) vs member 1")
    if changed == 0:
        print("WARNING: identical to member 1 -- members are not decorrelated")

    for d in docs:
        v = out[d["id"]]
        assert len(v) == len(d["text"]) and all(x in LABELS for x in v)

    sub_dir = os.path.join(ROOT, "submissions", args.sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    json_path = os.path.join(sub_dir, args.json_name)
    with open(json_path, "w") as f:
        json.dump(out, f)

    dist = Counter(x for v in out.values() for x in v)
    with open(os.path.join(sub_dir, "config.txt"), "w") as f:
        f.write(f"submission     {args.sub_name}\n")
        f.write(f"created        {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"method         majority vote, ties -> first listed\n")
        for c in args.rr_checkpoints:
            f.write(f"  member       {c}\n")
        f.write(f"base_checkpoint {args.base_checkpoint}\n")
        if args.note:
            f.write(f"note           {args.note}\n")
        f.write(f"\nvote changed {changed}/{n} ({100*changed/n:.2f}%)\n\n")
        for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
            f.write(f"  {k:26s} {v:6d}  {100*v/n:5.2f}%\n")

    with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname=args.json_name)

    print(f"\n{sub_dir}/ written")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {k:26s} {v:6d}  {100*v/n:5.2f}%")


if __name__ == "__main__":
    main()