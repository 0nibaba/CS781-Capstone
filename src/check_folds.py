"""
Are fold1 and fold2 comparable enough to cross-validate across?

Two risks:
  1. If the label distributions differ sharply, CV variance would reflect the
     split rather than the method being tested.
  2. Phase 1 found that adjacent documents in this corpus can be near
     duplicates. If fold1 and fold2 share material, CV would be optimistic —
     the same mistake the Phase 1 eval set made before contamination
     filtering.
"""

import os
import json
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")
TEST = os.path.join(ROOT, "test_data/Phase_2_LNER/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
          "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]


def load(path):
    with open(path) as f:
        return json.load(f)


def shingles(text, k=5, cap=6000):
    w = text.lower().split()[:cap]
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    tok = Tokenizer.from_file(TOKENIZER)

    folds = {}
    for f in ("fold1", "fold2"):
        folds[f] = (load(os.path.join(TRAIN, f"{f}-text.json")),
                    load(os.path.join(TRAIN, f"{f}-gold.json")))
    test = load(os.path.join(TEST, "fold3-text.json"))

    # ---- label balance ----
    print("=" * 66)
    print("label counts per fold  (macro-F1 weights all 12 equally)")
    print("=" * 66)
    counts = {}
    for f, (_, gold) in folds.items():
        c = Counter(e[2] for ents in gold.values() for e in ents)
        counts[f] = c

    print(f"  {'label':<12} {'fold1':>7} {'fold2':>7} {'ratio':>7}")
    print("  " + "-" * 36)
    worst = []
    for lab in LABELS:
        a, b = counts["fold1"].get(lab, 0), counts["fold2"].get(lab, 0)
        r = max(a, b) / max(1, min(a, b))
        worst.append((r, lab, a, b))
        flag = "  <-- imbalanced" if r > 2.0 else ""
        print(f"  {lab:<12} {a:>7} {b:>7} {r:>6.2f}x{flag}")
    print(f"  {'TOTAL':<12} {sum(counts['fold1'].values()):>7} "
          f"{sum(counts['fold2'].values()):>7}")

    thin = [(lab, a, b) for _, lab, a, b in worst if min(a, b) < 60]
    if thin:
        print("\n  labels with <60 instances in a fold (noisy per-fold F1):")
        for lab, a, b in thin:
            print(f"    {lab:<12} fold1 {a}, fold2 {b}")

    # ---- document length ----
    print("\n" + "=" * 66)
    print("document length per fold")
    print("=" * 66)
    for f, (docs, gold) in folds.items():
        toks = np.array([len(tok.encode(t).ids) for t in docs.values()])
        ents = np.array([len(gold[d]) for d in docs])
        print(f"  {f}: {len(docs)} docs | tokens median {int(np.median(toks)):,} "
              f"p90 {int(np.percentile(toks,90)):,} max {toks.max():,}")
        print(f"         entities/doc median {int(np.median(ents))} "
              f"total {ents.sum():,}")
    toks_t = np.array([len(tok.encode(t).ids) for t in test.values()])
    print(f"  fold3 (test): {len(test)} docs | tokens median "
          f"{int(np.median(toks_t)):,} max {toks_t.max():,}")

    # ---- near-duplicate leakage ----
    print("\n" + "=" * 66)
    print("cross-fold similarity (Phase 1 found near-duplicates in this corpus)")
    print("=" * 66)
    print("  computing shingles...", flush=True)
    sh = {f: {d: shingles(t) for d, t in folds[f][0].items()} for f in folds}
    sh_test = {d: shingles(t) for d, t in test.items()}

    def report(name, A, B, same=False):
        best = []
        for da, sa in A.items():
            for db, sb in B.items():
                if same and da >= db:
                    continue
                best.append((jaccard(sa, sb), da, db))
        best.sort(reverse=True)
        top = best[:5]
        vals = np.array([b[0] for b in best])
        print(f"\n  {name}: {len(best):,} pairs | "
              f"median {np.median(vals):.3f} p99 {np.percentile(vals,99):.3f} "
              f"max {vals.max():.3f}")
        over = (vals > 0.5).sum()
        print(f"    pairs above 0.5: {over}")
        for j, da, db in top[:3]:
            print(f"      {j:.3f}  {da} <-> {db}")

    report("fold1 vs fold2", sh["fold1"], sh["fold2"])
    report("within fold1", sh["fold1"], sh["fold1"], same=True)
    report("train vs test (fold3)",
           {**sh["fold1"], **sh["fold2"]}, sh_test)

    print("\n" + "=" * 66)
    print("reading")
    print("=" * 66)
    print("  label ratios near 1.0 and no high-similarity cross-fold pairs")
    print("  -> the given folds are safe to cross-validate across.")
    print("  high fold1-fold2 similarity -> CV would be optimistic.")
    print("  high train-test similarity -> the leaderboard will flatter us,")
    print("  the same way the Phase 1 eval set did before filtering.")


if __name__ == "__main__":
    main()