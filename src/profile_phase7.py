import os, json, argparse
from collections import Counter
import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = os.path.join(ROOT, "training_data/raw/Phase_7_PCR/Train")
DV = os.path.join(ROOT, "training_data/raw/Phase_7_PCR/Dev")
TE = os.path.join(ROOT, "test_data/Phase_7_PCR/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")


def load(p):
    with open(p) as f:
        return json.load(f)


def as_text(v):
    if isinstance(v, dict):
        v = v.get("text", v.get("document", ""))
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=TOKENIZER)
    args = p.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)

    splits = {
        "train": (os.path.join(TR, "train.json"),
                  os.path.join(TR, "train_candidates.json")),
        "dev": (os.path.join(DV, "dev.json"),
                os.path.join(DV, "dev_candidates.json")),
        "test": (os.path.join(TE, "test.json"),
                 os.path.join(TE, "test_candidates.json")),
    }

    pools = {}
    for name, (qp, cp) in splits.items():
        if not (os.path.exists(qp) and os.path.exists(cp)):
            print(f"{name}: MISSING")
            continue
        Q, C = load(qp), load(cp)
        qs = list(Q.items()) if isinstance(Q, dict) else [
            (x["id"], x) for x in Q]
        pools[name] = set(C.keys()) if isinstance(C, dict) else {
            x["id"] for x in C}

        print(f"\n=== {name}: {len(qs)} queries | {len(pools[name])} candidates ===")
        sample = qs[0][1]
        print(f"  query fields: "
              f"{sorted(sample.keys()) if isinstance(sample, dict) else type(sample)}")

        rel = []
        for qid, q in qs:
            if isinstance(q, dict) and q.get("relevant_candidates") is not None:
                rel.append(len(q["relevant_candidates"]))
        if rel:
            r = np.array(rel)
            print(f"  relevant/query: mean {r.mean():.2f} median "
                  f"{np.median(r):.0f} min {r.min()} max {r.max()}")
            print(f"    exactly 7: {100*(r==7).mean():.1f}%  "
                  f"fewer than 7: {100*(r<7).mean():.1f}%  "
                  f"more: {100*(r>7).mean():.1f}%")
            tp = np.minimum(r, 7)
            prec = tp.sum() / (7 * len(r))
            recl = tp.sum() / r.sum()
            print(f"  PERFECT-ranking ceiling: micro-P {prec:.4f} "
                  f"micro-R {recl:.4f} micro-F1 "
                  f"{2*prec*recl/(prec+recl):.4f}")
            print(f"  random-7 baseline micro-R ~ "
                  f"{7/len(pools[name]):.5f}")

        ql = np.array([len(tok.encode(as_text(q)).ids) for _, q in qs[:300]])
        print(f"  query tokens: median {np.median(ql):.0f} "
              f"p90 {np.percentile(ql,90):.0f} max {ql.max()}")
        citems = list(C.items()) if isinstance(C, dict) else [
            (x["id"], x) for x in C]
        cl = np.array([len(tok.encode(as_text(c)).ids)
                       for _, c in citems[:300]])
        print(f"  cand  tokens: median {np.median(cl):.0f} "
              f"p90 {np.percentile(cl,90):.0f} max {cl.max()}")
        print(f"  embedding cost: {len(pools[name])} candidates x 1024 tokens")

    if len(pools) > 1:
        print("\n=== pool overlap ===")
        ks = list(pools)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = pools[ks[i]], pools[ks[j]]
                print(f"  {ks[i]} n {ks[j]}: {len(a & b)} shared "
                      f"({100*len(a&b)/min(len(a),len(b)):.1f}% of smaller)")


if __name__ == "__main__":
    main()