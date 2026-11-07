import os
import json
from collections import Counter, defaultdict

from rr_data import LABELS, load_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def macro_scores(pred, gold, labels=LABELS):
    """pred, gold: {doc_id: [label, ...]} with matching lengths."""
    tp, fp, fn = Counter(), Counter(), Counter()
    n = correct = 0

    for did in gold:
        g, p = gold[did], pred.get(did)
        if p is None:
            raise KeyError(f"no prediction for {did}")
        if len(p) != len(g):
            raise ValueError(f"{did}: {len(p)} predictions for {len(g)} sentences")
        for a, b in zip(g, p):
            n += 1
            if a == b:
                tp[a] += 1
                correct += 1
            else:
                fn[a] += 1
                fp[b] += 1

    per = {}
    for lab in labels:
        if tp[lab] + fp[lab] + fn[lab] == 0:
            continue
        per[lab] = prf(tp[lab], fp[lab], fn[lab]) + (tp[lab], fp[lab], fn[lab])

    if not per:
        return 0.0, 0.0, 0.0, 0.0, {}
    mp = sum(v[0] for v in per.values()) / len(per)
    mr = sum(v[1] for v in per.values()) / len(per)
    mf = sum(v[2] for v in per.values()) / len(per)
    return mf, mp, mr, correct / max(1, n), per


def report(pred, gold, title=""):
    mf, mp, mr, acc, per = macro_scores(pred, gold)
    if title:
        print(f"\n{title}")
    print(f"  macro-F1 {mf:.4f}  (P {mp:.4f} / R {mr:.4f})   "
          f"accuracy {acc:.4f}   labels scored {len(per)}/{len(LABELS)}")
    print(f"  {'label':<24} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'tp':>6} {'fp':>6} {'fn':>6}")
    print("  " + "-" * 68)
    for lab in sorted(per, key=lambda l: per[l][2]):
        p, r, f, t, fpv, fnv = per[lab]
        print(f"  {lab:<24} {p:>7.4f} {r:>7.4f} {f:>7.4f} "
              f"{t:>6} {fpv:>6} {fnv:>6}")
    return mf


def baselines(train_docs, eval_docs):
    gold = {d["id"]: d["labels"] for d in eval_docs}
    counts = Counter(l for d in train_docs for l in d["labels"])
    majority = counts.most_common(1)[0][0]

    out = {}

    out["majority class"] = {d["id"]: [majority] * len(d["text"])
                             for d in eval_docs}

    first = defaultdict(Counter)
    for d in train_docs:
        first[d["domain"]][d["labels"][0]] += 1
    out["majority, per domain"] = {
        d["id"]: [counts_for_domain(train_docs, d["domain"])] * len(d["text"])
        for d in eval_docs}

    out["copy previous gold"] = {
        d["id"]: [d["labels"][0]] + d["labels"][:-1] for d in eval_docs}

    return out, gold


def counts_for_domain(train_docs, domain):
    c = Counter(l for d in train_docs if d["domain"] == domain
                for l in d["labels"])
    return c.most_common(1)[0][0]


if __name__ == "__main__":
    train = load_split("train")
    dev = load_split("dev")

    print(f"train {len(train)} documents, dev {len(dev)} documents")
    print(f"dev sentences: {sum(len(d['text']) for d in dev):,}")

    dev_labels = Counter(l for d in dev for l in d["labels"])
    absent = [l for l in LABELS if dev_labels[l] == 0]
    print(f"\nlabels with ZERO dev examples: {absent}")
    print("  dev macro-F1 is averaged over fewer labels than test will be, so")
    print("  it is not directly comparable to the leaderboard")

    runs, gold = baselines(train, dev)
    print("\n" + "=" * 72)
    print("trivial baselines on dev (both domains pooled)")
    print("=" * 72)
    for name, pred in runs.items():
        mf, mp, mr, acc, per = macro_scores(pred, gold)
        print(f"  {name:<24} macro-F1 {mf:.4f}   accuracy {acc:.4f}   "
              f"labels scored {len(per)}")

    print("\n  'copy previous gold' is not achievable — it reads the answer —")
    print("  but its accuracy is the share of the task that is pure continuity,")
    print("  and its macro-F1 is what continuity alone is worth under this metric.")

    print("\n" + "=" * 72)
    print("per domain")
    print("=" * 72)
    for dom in ("CL", "IT"):
        sub = [d for d in dev if d["domain"] == dom]
        if not sub:
            continue
        g = {d["id"]: d["labels"] for d in sub}
        for name, pred in runs.items():
            p = {k: v for k, v in pred.items() if k in g}
            mf, _, _, acc, per = macro_scores(p, g)
            print(f"  {dom}  {name:<24} macro-F1 {mf:.4f}   acc {acc:.4f}")

    report(runs["copy previous gold"], gold, "copy previous gold, per label")