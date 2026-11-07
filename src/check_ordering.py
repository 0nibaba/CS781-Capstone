import os
import json
import random
import numpy as np
import ijson
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "training_data/raw/Phase_1_Pretraining/Train")

FILES = {
    "acts":     "train_acts.json",
    "high":     "train_high_courts.json",
    "supreme":  "train_supreme_court.json",
    "tribunal": "train_tribunals.json",
}

N_DOCS = 2000
N_RANDOM = 2000
SEED = 1899
MIN_WORDS = 50


def stream(path, n):
    out = []
    with open(path, "rb") as f:
        for rec in ijson.items(f, "item"):
            if not isinstance(rec, dict):
                continue
            t = rec.get("text")
            if not t or len(t.split()) < MIN_WORDS:
                continue
            out.append(rec)
            if len(out) >= n:
                break
    return out


def content_words(text, cap=3000):
    ws = text.lower().split()[:cap]
    return {w.strip(".,;:()[]\"'") for w in ws
            if len(w) >= 5 and w.isalpha()}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    rng = random.Random(SEED)

    for src, fname in FILES.items():
        path = os.path.join(RAW, fname)
        print(f"\n{'=' * 70}\n{src}\n{'=' * 70}")

        recs = stream(path, N_DOCS)
        print(f"read {len(recs)} documents in file order")

        keys = sorted(set().union(*(r.keys() for r in recs)))
        print(f"fields: {keys}")

        for field in ("court", "court_id", "bench", "year", "date", "tribunal", "category", "filepath"):
            vals = [r.get(field) for r in recs]
            if all(v is None for v in vals):
                continue

            if field == "filepath":
                vals = [os.path.dirname(v) if v else None for v in vals]

            adj = [vals[i] == vals[i + 1] for i in range(len(vals) - 1)
                   if vals[i] is not None and vals[i + 1] is not None]
            idx = [i for i, v in enumerate(vals) if v is not None]
            ran = [vals[rng.choice(idx)] == vals[rng.choice(idx)]
                   for _ in range(N_RANDOM)] if idx else []

            if adj and ran:
                a, r = 100 * np.mean(adj), 100 * np.mean(ran)
                ratio = a / r if r > 0 else float("inf")
                flag = "  <== NON-RANDOM" if a > 2 * r and a > 10 else ""
                label = field + " (dir)" if field == "filepath" else field
                print(f"  {label:<14} adjacent {a:5.1f}%  random {r:5.1f}%  "
                      f"ratio {ratio:5.1f}x{flag}")

        sets = [content_words(r["text"]) for r in recs]
        adj_j = [jaccard(sets[i], sets[i + 1]) for i in range(len(sets) - 1)]
        ran_j = [jaccard(sets[rng.randrange(len(sets))],
                         sets[rng.randrange(len(sets))])
                 for _ in range(N_RANDOM)]

        am, rm = float(np.mean(adj_j)), float(np.mean(ran_j))
        print(f"  {'lexical':<14} adjacent {am:.4f}   random {rm:.4f}   "
              f"ratio {am / rm if rm else float('inf'):5.2f}x")
        print(f"  {'':14} adjacent median {np.median(adj_j):.4f}, "
              f"p90 {np.percentile(adj_j, 90):.4f}")

        hi = 100 * np.mean(np.array(adj_j) > np.percentile(ran_j, 99))
        print(f"  {'':14} {hi:.1f}% of adjacent pairs exceed the 99th "
              f"percentile of random pairs")

        order = np.argsort(adj_j)[::-1][:3]
        print(f"\n  most similar adjacent pairs:")
        for i in order:
            i = int(i)
            for j in (i, i + 1):
                r = recs[j]
                label = (r.get("title") or r.get("tribunal")
                         or r.get("court") or "")[:70]
                meta = " ".join(str(r.get(k)) for k in ("year", "date", "tribunal")
                                if r.get(k))
                print(f"    [{j}] {label}   {meta}")
            print(f"    jaccard {adj_j[i]:.3f}\n")

    print(f"\n{'=' * 70}")
    print("Interpretation: adjacency >> random on metadata or lexical overlap")
    print("means the raw file preserves an ordering in which neighbouring")
    print("documents are related. Where that holds, packed documents are NOT")
    print("independent and masking removes real context rather than noise.")
    print("The CONTRAST between sources is what matters — tribunal alone")
    print("showing this would explain the tribunal-only masking penalty.")


if __name__ == "__main__":
    main()