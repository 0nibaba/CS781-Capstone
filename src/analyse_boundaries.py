"""
Why are 27% of entities found with the right label but the wrong span?

The deltas are dominated by 'start late, end correct' (+6/+0, +8/+0, +2/+0),
which looks systematic rather than random. Three candidate causes:

  A. the model genuinely starts the entity at the wrong token
  B. the gold spans include a prefix the model does not learn to include
     (a title, an honorific, an article)
  C. the decode is dropping leading tokens — a bug in bio_to_spans or in the
     trim convention

(C) would be the cheapest to fix and the most embarrassing to miss, so it is
tested first: round-tripping the GOLD spans through the same decode path
should reproduce them, and test_alignment measured that at 98.18%. If the
model's errors have the same shape as the few round-trip failures, it is a
decoding problem; if not, it is a learning problem.
"""

import os
import json
import argparse
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold_fold", required=True, choices=["fold1", "fold2"])
    args = ap.parse_args()

    pred = json.load(open(args.predictions))
    gold = {k: [[e[0], e[1], e[2]] for e in v] for k, v in
            json.load(open(os.path.join(TRAIN, f"{args.gold_fold}-gold.json"))).items()}
    texts = json.load(open(os.path.join(TRAIN, f"{args.gold_fold}-text.json")))

    # collect boundary-only errors
    cases = []
    for did in gold:
        P = [list(e[:3]) for e in pred.get(did, [])]
        for g in gold[did]:
            if any(p == g for p in P):
                continue
            ov = [p for p in P if overlaps(p, g) and p[2] == g[2]]
            if not ov:
                continue
            p = max(ov, key=lambda q: min(q[1], g[1]) - max(q[0], g[0]))
            if p[0] != g[0] or p[1] != g[1]:
                cases.append((did, g, p))

    print(f"{len(cases):,} boundary-only errors\n")

    # ---- what is in the missing prefix? ----
    print("=" * 66)
    print("text the model OMITTED at the start (gold starts earlier)")
    print("=" * 66)
    prefixes = Counter()
    suffixes = Counter()
    for did, g, p in cases:
        t = texts[did]
        if p[0] > g[0]:
            prefixes[t[g[0]:p[0]]] += 1
        if p[1] < g[1]:
            suffixes[t[p[1]:g[1]]] += 1
    print(f"  {sum(prefixes.values()):,} cases start too late")
    for s, n in prefixes.most_common(15):
        print(f"    {n:>4}  {s!r}")

    print(f"\n  {sum(suffixes.values()):,} cases end too early")
    for s, n in suffixes.most_common(10):
        print(f"    {n:>4}  {s!r}")

    # ---- text the model ADDED ----
    print("\n" + "=" * 66)
    print("text the model INCLUDED that gold does not")
    print("=" * 66)
    extra_pre, extra_post = Counter(), Counter()
    for did, g, p in cases:
        t = texts[did]
        if p[0] < g[0]:
            extra_pre[t[p[0]:g[0]]] += 1
        if p[1] > g[1]:
            extra_post[t[g[1]:p[1]]] += 1
    print(f"  {sum(extra_pre.values()):,} start too early:")
    for s, n in extra_pre.most_common(8):
        print(f"    {n:>4}  {s!r}")
    print(f"\n  {sum(extra_post.values()):,} end too late:")
    for s, n in extra_post.most_common(8):
        print(f"    {n:>4}  {s!r}")

    # ---- per label, which direction ----
    print("\n" + "=" * 66)
    print("direction by label")
    print("=" * 66)
    by_lab = defaultdict(Counter)
    for did, g, p in cases:
        by_lab[g[2]]["start late" if p[0] > g[0] else
                      "start early" if p[0] < g[0] else "start ok"] += 1
        by_lab[g[2]]["end early" if p[1] < g[1] else
                     "end late" if p[1] > g[1] else "end ok"] += 1
    print(f"  {'label':<12} {'st.late':>8} {'st.early':>9} "
          f"{'end.early':>10} {'end.late':>9}")
    for lab in sorted(by_lab, key=lambda l: -sum(by_lab[l].values())):
        c = by_lab[lab]
        print(f"  {lab:<12} {c['start late']:>8} {c['start early']:>9} "
              f"{c['end early']:>10} {c['end late']:>9}")

    # ---- worked examples for the worst label ----
    worst = max(by_lab, key=lambda l: sum(by_lab[l].values()))
    print("\n" + "=" * 66)
    print(f"examples: {worst}")
    print("=" * 66)
    shown = 0
    for did, g, p in cases:
        if g[2] != worst or shown >= 8:
            continue
        t = texts[did]
        print(f"\n  gold [{g[0]},{g[1]}] {t[g[0]:g[1]]!r}")
        print(f"  pred [{p[0]},{p[1]}] {t[p[0]:p[1]]!r}")
        shown += 1

    # ---- is the gold itself recoverable? (rules out a decode bug) ----
    print("\n" + "=" * 66)
    print("control: do these same gold spans survive a clean round trip?")
    print("=" * 66)
    from tokenizers import Tokenizer
    from bio_tagging import spans_to_bio, bio_to_spans
    tokzr = Tokenizer.from_file(os.path.join(
        ROOT, "checkpoints/tokenizers/legal_32000.json"))

    affected = {did for did, _, _ in cases}
    ok = bad = 0
    for did in list(affected)[:20]:
        t = texts[did]
        enc = tokzr.encode(t)
        tags, _ = spans_to_bio(gold[did], enc.offsets, len(enc.ids))
        rt = {tuple(x) for x in bio_to_spans(tags, enc.offsets, t)}
        for _, g, _ in [c for c in cases if c[0] == did]:
            if tuple(g) in rt:
                ok += 1
            else:
                bad += 1
    print(f"  of the boundary-error gold spans in 20 documents, "
          f"{ok} round-trip exactly and {bad} do not")
    print("  if nearly all round-trip, the tokenizer CAN represent them and")
    print("  this is a learning problem, not a decoding bug")


if __name__ == "__main__":
    main()