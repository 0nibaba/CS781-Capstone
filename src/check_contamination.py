import os
import json
import numpy as np
import ijson
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "training_data/raw/Phase_1_Pretraining/Train")
EVAL_SET = os.path.join(ROOT, "test_data/eval_set.json")
HELDOUT_HASHES = os.path.join(ROOT, "test_data/heldout_hashes.json")

FILES = {
    "acts":     "train_acts.json",
    "high":     "train_high_courts.json",
    "supreme":  "train_supreme_court.json",
    "tribunal": "train_tribunals.json",
}

NEIGHBOURHOOD = 10
SHINGLE = 5
CAP = 4000


def shingles(text, k=SHINGLE, cap=CAP):
    w = text.lower().split()[:cap]
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def stream(path):
    with open(path, "rb") as f:
        for rec in ijson.items(f, "item"):
            if isinstance(rec, dict) and rec.get("text"):
                yield rec


def main():
    import hashlib

    with open(EVAL_SET) as f:
        eval_set = json.load(f)
    print(f"eval set: {len(eval_set)} examples")

    with open(HELDOUT_HASHES) as f:
        d = json.load(f)
    heldout = {h for hs in d.values() for h in hs} if isinstance(d, dict) else set(d)
    print(f"held-out hashes: {len(heldout):,}\n")

    eval_by_src = defaultdict(list)
    for k, v in eval_set.items():
        eval_by_src[v.get("source", "?")].append(
            (k, shingles(v["prompt"] + " " + v["reference"])))

    best = {k: (0.0, "") for k in eval_set}

    for src, fname in FILES.items():
        if src not in eval_by_src:
            continue
        path = os.path.join(RAW, fname)
        targets = eval_by_src[src]
        print(f"[{src}] {len(targets)} eval windows, scanning {fname}...",
              flush=True)

        buf = []
        n_heldout_found = 0
        n = 0

        for rec in stream(path):
            text = rec["text"]
            h = hashlib.md5(text.encode()).hexdigest()
            buf.append((h, text, rec.get("title", "")))
            if len(buf) > 2 * NEIGHBOURHOOD + 1:
                buf.pop(0)
            n += 1

            if len(buf) == 2 * NEIGHBOURHOOD + 1:
                centre = buf[NEIGHBOURHOOD]
                if centre[0] in heldout:
                    n_heldout_found += 1
                    for i, (nh, ntext, ntitle) in enumerate(buf):
                        if i == NEIGHBOURHOOD or nh in heldout:
                            continue
                        nsh = shingles(ntext)
                        for k, esh in targets:
                            j = jaccard(esh, nsh)
                            if j > best[k][0]:
                                best[k] = (j, f"{src} offset {i-NEIGHBOURHOOD:+d}: "
                                              f"{ntitle[:45]}")

            if n % 100_000 == 0:
                print(f"  {n:,} docs, {n_heldout_found} held-out found",
                      flush=True)

        print(f"  {n:,} documents, {n_heldout_found} held-out documents located\n")

    scores = {k: best[k][0] for k in eval_set}
    arr = np.array(list(scores.values()))

    print("=" * 78)
    print("best Jaccard against a TRAINING neighbour of a held-out document")
    print("=" * 78)
    for t in (0.9, 0.8, 0.7, 0.5, 0.3):
        c = int((arr >= t).sum())
        print(f"  >= {t}: {c:4d} / {len(arr)}  ({100*c/len(arr):5.1f}%)")

    print(f"\n  median {np.median(arr):.3f}  p90 {np.percentile(arr, 90):.3f}  "
          f"max {arr.max():.3f}")

    by_src = defaultdict(list)
    for k, v in eval_set.items():
        by_src[v.get("source", "?")].append(scores[k])
    print("\n  by source:")
    for s in sorted(by_src):
        a = np.array(by_src[s])
        print(f"    {s:<10} median {np.median(a):.3f}  "
              f">=0.5: {int((a >= 0.5).sum()):3d}/{len(a)}  "
              f"max {a.max():.3f}")

    print("\n  highest-scoring eval examples:")
    for k in sorted(scores, key=scores.get, reverse=True)[:10]:
        print(f"    {k}  {scores[k]:.3f}  <- {best[k][1]}")

    clean = sorted(k for k in eval_set if scores[k] < 0.5)
    out = os.path.join(ROOT, "logs", "contamination.json")
    with open(out, "w") as f:
        json.dump({"scores": scores,
                   "matches": {k: best[k][1] for k in eval_set},
                   "clean_ids": clean}, f, indent=2)

    print(f"\nclean subset (best match < 0.5): {len(clean)} / {len(eval_set)}")
    print(f"saved -> {out}")
    print("\nNext: re-score arm A and arm B on the clean subset only. If the")
    print("0.036 gap holds there the finding survives; if it collapses, the")
    print("headline result was an artifact of memorised near-duplicates.")


if __name__ == "__main__":
    main()