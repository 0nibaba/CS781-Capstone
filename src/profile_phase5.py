"""What LSI actually looks like, before any modelling decisions."""
import os, json, argparse
from collections import Counter
import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training_data/raw/Phase_5_LSI/Train")
TEST_DIR = os.path.join(ROOT, "test_data/Phase_5_LSI/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")


def load(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "text": v}
             for k, v in d.items()]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=TOKENIZER)
    args = p.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)

    splits = {"train": os.path.join(TRAIN_DIR, "train.json"),
              "dev": os.path.join(TRAIN_DIR, "dev.json"),
              "test": os.path.join(TEST_DIR, "test.json")}

    label_sets = {}
    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"{name}: MISSING {path}")
            continue
        docs = load(path)
        print(f"\n=== {name}: {len(docs)} documents ===")
        print(f"  keys: {sorted(docs[0].keys())}")

        labs = [d["labels"] for d in docs if d.get("labels") is not None]
        if labs:
            flat = Counter(l for ls in labs for l in ls)
            label_sets[name] = set(flat)
            n_per = np.array([len(ls) for ls in labs])
            print(f"  distinct statutes: {len(flat)}  "
                  f"(min id {min(flat)}, max id {max(flat)})")
            print(f"  labels/doc: mean {n_per.mean():.2f} median "
                  f"{np.median(n_per):.0f} min {n_per.min()} max {n_per.max()}")
            print(f"  docs with 1 label: {100*(n_per==1).mean():.1f}%  "
                  f"with >3: {100*(n_per>3).mean():.1f}%")

            freq = np.array(sorted(flat.values(), reverse=True))
            tot = freq.sum()
            print(f"  most common statute: {freq[0]} docs "
                  f"({100*freq[0]/len(docs):.1f}% of docs)")
            print(f"  head 10 statutes cover {100*freq[:10].sum()/tot:.1f}% "
                  f"of all label occurrences")
            for k in (1, 2, 5, 10, 20):
                print(f"    statutes appearing <= {k:2d} times: "
                      f"{(freq <= k).sum()} of {len(freq)}")
            # a rare statute is unreachable if macro-F1 averages over all of them
            print(f"  macro-F1 ceiling if every statute with <5 examples "
                  f"scores 0: {1 - (freq < 5).sum()/len(freq):.3f}")

        lens = np.array([len(tok.encode(d["text"]).ids) for d in docs])
        print(f"  tokens: mean {lens.mean():.0f} median {np.median(lens):.0f} "
              f"max {lens.max()}")
        for q in (50, 90, 95, 99):
            print(f"    p{q}: {np.percentile(lens, q):.0f}")
        for w in (512, 1024, 2048):
            print(f"    fits in {w}: {100*(lens<=w).mean():.1f}%  "
                  f"chunks at {w}: mean {np.ceil(lens/w).mean():.1f}")

    if "train" in label_sets and "dev" in label_sets:
        tr, dv = label_sets["train"], label_sets["dev"]
        print(f"\n=== label coverage ===")
        print(f"  train {len(tr)} | dev {len(dv)}")
        print(f"  in dev but not train: {len(dv - tr)}  {sorted(dv - tr)[:20]}")
        print(f"  in train but not dev: {len(tr - dv)}")


if __name__ == "__main__":
    main()