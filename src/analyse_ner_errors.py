import os
import json
import argparse
from collections import defaultdict, Counter

from bio_tagging import LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def categorise(pred, gold):
    """Returns counters and example lists."""
    cat = Counter()
    confusion = defaultdict(Counter)
    boundary = defaultdict(Counter)
    examples = defaultdict(list)

    for did in set(pred) | set(gold):
        P = [list(e[:3]) for e in pred.get(did, [])]
        G = [list(e[:3]) for e in gold.get(did, [])]
        used_p = set()

        for g in G:
            exact = [i for i, p in enumerate(P)
                     if p[0] == g[0] and p[1] == g[1] and p[2] == g[2]]
            if exact:
                cat["exact"] += 1
                used_p.add(exact[0])
                continue

            ov = [i for i, p in enumerate(P) if overlaps(p, g) and i not in used_p]
            if not ov:
                cat["missed"] += 1
                examples["missed"].append((did, g))
                continue

            i = max(ov, key=lambda k: min(P[k][1], g[1]) - max(P[k][0], g[0]))
            p = P[i]
            used_p.add(i)

            if p[2] == g[2]:
                cat["boundary"] += 1
                boundary[g[2]][(p[0] - g[0], p[1] - g[1])] += 1
                examples["boundary"].append((did, g, p))
            elif p[0] == g[0] and p[1] == g[1]:
                cat["label"] += 1
                confusion[g[2]][p[2]] += 1
                examples["label"].append((did, g, p))
            else:
                cat["both"] += 1
                confusion[g[2]][p[2]] += 1
                examples["both"].append((did, g, p))

        for i, p in enumerate(P):
            if i not in used_p:
                cat["spurious"] += 1
                examples["spurious"].append((did, p))

    return cat, confusion, boundary, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold_fold", required=True, choices=["fold1", "fold2"])
    args = ap.parse_args()

    with open(args.predictions) as f:
        pred = json.load(f)
    with open(os.path.join(TRAIN, f"{args.gold_fold}-gold.json")) as f:
        gold = {k: [[e[0], e[1], e[2]] for e in v]
                for k, v in json.load(f).items()}
    with open(os.path.join(TRAIN, f"{args.gold_fold}-text.json")) as f:
        texts = json.load(f)

    cat, confusion, boundary, examples = categorise(pred, gold)

    n_gold = sum(len(v) for v in gold.values())
    n_pred = sum(len(v) for v in pred.values())
    print(f"{n_gold:,} gold entities, {n_pred:,} predicted "
          f"({n_pred/n_gold:.2f}x)\n")

    print("=" * 62)
    print("error breakdown")
    print("=" * 62)
    order = ["exact", "boundary", "label", "both", "missed", "spurious"]
    desc = {
        "exact": "correct",
        "boundary": "right label, wrong span",
        "label": "right span, wrong label",
        "both": "overlapping, both wrong",
        "missed": "no overlapping prediction",
        "spurious": "predicted where nothing is",
    }
    for k in order:
        base = n_pred if k == "spurious" else n_gold
        print(f"  {k:<10} {cat[k]:>7,}  {100*cat[k]/base:>5.1f}%  {desc[k]}")

    print("\n" + "=" * 62)
    print("label confusion  (gold -> predicted, overlapping spans only)")
    print("=" * 62)
    present = [l for l in LABELS if confusion[l]]
    if present:
        print(f"  {'gold':<12} " + " ".join(f"{l[:5]:>6}" for l in LABELS))
        for g in present:
            row = " ".join(f"{confusion[g].get(p, 0):>6}" for p in LABELS)
            print(f"  {g:<12} {row}")

        print("\n  worst confusions:")
        pairs = [(n, g, p) for g in confusion for p, n in confusion[g].items()]
        for n, g, p in sorted(pairs, reverse=True)[:8]:
            print(f"    {g:>10} predicted as {p:<10} {n:>4}")

    print("\n" + "=" * 62)
    print("boundary errors  (right label, wrong span)")
    print("=" * 62)
    allb = Counter()
    for lab, c in boundary.items():
        allb.update(c)
    for (ds, de), n in allb.most_common(8):
        print(f"  start {ds:+3d}  end {de:+3d}   {n:>4}")
    print("\n  by label:")
    for lab in sorted(boundary, key=lambda l: -sum(boundary[l].values()))[:6]:
        print(f"    {lab:<12} {sum(boundary[lab].values()):>4}")

    print("\n" + "=" * 62)
    print("examples")
    print("=" * 62)
    for kind in ("label", "spurious", "missed"):
        print(f"\n  --- {kind} ---")
        for item in examples[kind][:6]:
            did = item[0]
            text = texts.get(did, "")
            if kind == "spurious":
                p = item[1]
                print(f"    predicted {p[2]:<10} {text[p[0]:p[1]]!r}")
                print(f"      context ...{text[max(0,p[0]-40):p[0]]!r} >>> "
                      f"{text[p[1]:p[1]+30]!r}")
            elif kind == "missed":
                g = item[1]
                print(f"    gold {g[2]:<10} {text[g[0]:g[1]]!r}")
            else:
                g, p = item[1], item[2]
                print(f"    gold {g[2]:<10} pred {p[2]:<10} "
                      f"{text[g[0]:g[1]]!r}")
                print(f"      context ...{text[max(0,g[0]-40):g[0]]!r} >>> "
                      f"{text[g[1]:g[1]+30]!r}")


if __name__ == "__main__":
    main()