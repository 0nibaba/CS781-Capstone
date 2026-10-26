"""Score the majority vote on dev, using each run's saved best-epoch predictions."""
import os, json, argparse
from collections import Counter
from rr_data import load_split
from rr_metrics import macro_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="best-first")
    args = p.parse_args()

    docs = load_split("dev", ("CL", "IT"))
    gold = {d["id"]: d["labels"] for d in docs}
    seen = {l for v in gold.values() for l in v}

    def f1(v):
        return v["f1"] if isinstance(v, dict) else v[2]

    def score(pred):
        mf, mp, mr, acc, per = macro_scores(pred, gold)
        vis = [f1(per[l]) for l in seen if l in per]
        return sum(vis) / len(vis), acc

    members = []
    for r in args.runs:
        path = os.path.join(ROOT, "checkpoints/rr", r, "predictions.json")
        with open(path) as f:
            m = json.load(f)
        members.append(m)
        mf, acc = score(m)
        print(f"  {r:22s} macro {mf:.4f}  acc {acc:.4f}")

    out, changed = {}, 0
    for did, g in gold.items():
        seq = []
        for i in range(len(g)):
            votes = [m[did][i] for m in members]
            top = Counter(votes).most_common()
            best = max(c for _, c in top)
            tied = {l for l, c in top if c == best}
            pick = next(v for v in votes if v in tied)
            seq.append(pick)
            changed += pick != votes[0]
        out[did] = seq

    n = sum(len(v) for v in gold.values())
    mf, acc = score(out)
    print(f"\n  {'VOTE':22s} macro {mf:.4f}  acc {acc:.4f}"
          f"   ({changed}/{n} = {100*changed/n:.2f}% changed)")


if __name__ == "__main__":
    main()