"""
Document-level label consistency.

The role analysis found that within a single document, a surface string NEVER
takes two roles: 0 of 745 document-string pairs. The model is free to violate
that and may well do so.

This measures the model's inconsistency and then applies majority voting:
every predicted span with the same surface string in the same document is
forced to the most-predicted label. A pure decode-time change, no retraining —
the same shape of gain as the bridge-3 fix (+0.10 macro-F1).

Measurement comes first: if the model is already consistent, voting does
nothing and the idea should be dropped rather than implemented.
"""

import os
import json
import argparse
from collections import defaultdict, Counter

from tokenizers import Tokenizer

from ner_data import load_fold, TOKENIZER
from ner_metrics import strict_scores, report
from decode_variants import decode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BEST_DECODE = dict(lenient=False, bridge=3, min_chars=3)


def key(text, span):
    return text[span[0]:span[1]].strip().lower()


def measure(pred, docs):
    """How often does the model give one string two labels in one document?"""
    incons = 0
    total = 0
    spread = Counter()
    examples = []
    for did, spans in pred.items():
        groups = defaultdict(Counter)
        for s in spans:
            groups[key(docs[did], s)][s[2]] += 1
        for k, c in groups.items():
            total += 1
            if len(c) > 1:
                incons += 1
                spread[len(c)] += 1
                if len(examples) < 8:
                    examples.append((did, k, dict(c)))
    return incons, total, spread, examples


def vote(pred, docs, min_count=1):
    """Force every mention of a string in a document to the majority label.
    Ties go to the label with the longest total span, as a deterministic
    tiebreak rather than dict order."""
    out = {}
    for did, spans in pred.items():
        groups = defaultdict(list)
        for s in spans:
            groups[key(docs[did], s)].append(s)

        new = []
        for k, grp in groups.items():
            if len(grp) < min_count:
                new.extend(grp)
                continue
            c = Counter(s[2] for s in grp)
            top = max(c.items(),
                      key=lambda kv: (kv[1],
                                      sum(s[1] - s[0] for s in grp
                                          if s[2] == kv[0])))[0]
            new.extend([[s[0], s[1], top] for s in grp])
        out[did] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--fold", required=True, choices=["fold1", "fold2"])
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOKENIZER)
    docs, gold = load_fold(args.fold)
    gold = {k: [[e[0], e[1], e[2]] for e in v] for k, v in gold.items()}
    saved = json.load(open(args.tags))
    offsets = {d: tok.encode(t).offsets for d, t in docs.items()}

    pred = {d: decode(saved[d]["tags"], offsets[d], docs[d], **BEST_DECODE)
            for d in docs if d in saved}

    # ---- is the model inconsistent at all? ----
    incons, total, spread, examples = measure(pred, docs)
    print("=" * 66)
    print("model inconsistency")
    print("=" * 66)
    print(f"  {incons} of {total} document-string groups ({100*incons/total:.1f}%) "
          f"get more than one label")
    print(f"  gold does this 0 of 745 times (0.0%)")
    if spread:
        print(f"  labels per inconsistent group: {dict(spread)}")
    if not incons:
        print("\n  the model is already consistent — voting cannot help.")
        return
    print("\n  examples:")
    for did, k, c in examples:
        print(f"    {k[:40]!r:<44} {c}")

    # ---- does voting help? ----
    print("\n" + "=" * 66)
    print("effect of majority voting")
    print("=" * 66)
    mf0 = strict_scores(pred, gold)[0]
    print(f"  {'variant':<28} {'macroF1':>8} {'delta':>8}")
    print(f"  {'no voting':<28} {mf0:>8.4f}")

    best = (None, mf0, "no voting")
    for mc in (1, 2, 3):
        v = vote(pred, docs, min_count=mc)
        mf = strict_scores(v, gold)[0]
        name = f"vote (groups of >= {mc})"
        print(f"  {name:<28} {mf:>8.4f} {mf - mf0:>+8.4f}")
        if mf > best[1]:
            best = (v, mf, name)

    if best[0] is None:
        print("\n  voting does not help on this fold")
        return

    print(f"\n  best: {best[2]}  {best[1]:.4f} ({best[1]-mf0:+.4f})")
    report(best[0], gold, "per label, after voting")

    # which labels moved?
    _, _, _, per0, _ = strict_scores(pred, gold)
    _, _, _, per1, _ = strict_scores(best[0], gold)
    print(f"\n  {'label':<12} {'before':>8} {'after':>8} {'delta':>8}")
    for lab in sorted(per0, key=lambda l: per1.get(l, (0,0,0))[2]
                      - per0[l][2]):
        a, b = per0[lab][2], per1.get(lab, (0, 0, 0))[2]
        print(f"  {lab:<12} {a:>8.4f} {b:>8.4f} {b-a:>+8.4f}")


if __name__ == "__main__":
    main()