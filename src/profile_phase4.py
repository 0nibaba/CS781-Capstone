"""What CJPE actually looks like, before any modelling decisions."""
import os, json, argparse
from collections import Counter
import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training_data/raw/Phase_4_CJPE/Train")
TEST_DIR = os.path.join(ROOT, "test_data/Phase_4_CJPE/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")


def load(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):               # {id: {...}} or {id: text}
        out = []
        for k, v in d.items():
            out.append({"id": k, **v} if isinstance(v, dict)
                       else {"id": k, "text": v})
        return out
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=TOKENIZER)
    args = p.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)

    splits = {"train": os.path.join(TRAIN_DIR, "train.json"),
              "dev": os.path.join(TRAIN_DIR, "dev.json"),
              "test": os.path.join(TEST_DIR, "test.json")}

    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"{name}: MISSING {path}")
            continue
        docs = load(path)
        print(f"\n=== {name}: {len(docs)} cases ===")
        print(f"  keys: {sorted(docs[0].keys())}")

        labs = [d.get("label") for d in docs if d.get("label") is not None]
        if labs:
            c = Counter(labs)
            n = sum(c.values())
            for k, v in sorted(c.items()):
                print(f"  label {k}: {v} ({100*v/n:.1f}%)")
            # a majority-class predictor's macro-F1, the floor to beat
            maj = max(c.values()) / n
            print(f"  majority-class accuracy {maj:.4f}, "
                  f"macro-F1 {2*maj/(1+maj)/2:.4f}")

        lens = np.array([len(tok.encode(d["text"]).ids) for d in docs])
        print(f"  tokens: mean {lens.mean():.0f}  median "
              f"{np.median(lens):.0f}  min {lens.min()}  max {lens.max()}")
        for q in (50, 75, 90, 95, 99):
            print(f"    p{q}: {np.percentile(lens, q):.0f}")
        for w in (512, 1024, 2048, 4096):
            print(f"    fits in {w:5d} tokens: "
                  f"{100*(lens <= w).mean():5.1f}%   "
                  f"chunks needed at {w}: mean "
                  f"{np.ceil(lens/w).mean():.1f}")

        # does label correlate with length? a trivial shortcut if so
        if labs:
            for k in sorted(set(labs)):
                sel = lens[[i for i, d in enumerate(docs)
                            if d.get("label") == k]]
                print(f"  label {k}: median {np.median(sel):.0f} tokens")


if __name__ == "__main__":
    main()