"""What the summarization data looks like, before any modelling decisions."""
import os, json, argparse
import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training_data/raw/Phase_6_Summ/Train")
TEST_DIR = os.path.join(ROOT, "test_data/Phase_6_Summ/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")


def load(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "document": v}
             for k, v in d.items()]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=TOKENIZER)
    args = p.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)

    for name, path in (("train", os.path.join(TRAIN_DIR, "train.json")),
                       ("test", os.path.join(TEST_DIR, "test.json"))):
        if not os.path.exists(path):
            print(f"{name}: MISSING {path}")
            continue
        docs = load(path)
        print(f"\n=== {name}: {len(docs)} examples ===")
        print(f"  keys: {sorted(docs[0].keys())}")

        dl = np.array([len(tok.encode(d["document"]).ids) for d in docs])
        print(f"  document tokens: mean {dl.mean():.0f} median "
              f"{np.median(dl):.0f} max {dl.max()}")
        for q in (50, 75, 90, 95, 99):
            print(f"    p{q}: {np.percentile(dl, q):.0f}")
        for w in (512, 1024, 2048):
            print(f"    fits in {w}: {100*(dl<=w).mean():5.1f}%  "
                  f"chunks at {w}: mean {np.ceil(dl/w).mean():.1f}")

        if docs[0].get("summary") is not None:
            sl = np.array([len(tok.encode(d["summary"]).ids) for d in docs])
            print(f"  summary tokens: mean {sl.mean():.0f} median "
                  f"{np.median(sl):.0f} min {sl.min()} max {sl.max()}")
            for q in (50, 90, 95, 99):
                print(f"    p{q}: {np.percentile(sl, q):.0f}")
            print(f"  compression doc/summary: median "
                  f"{np.median(dl/np.clip(sl,1,None)):.1f}x")
            # how much of a 1024 budget is left for the source?
            for b in (256, 512, 768):
                print(f"    summaries under {b} tokens: "
                      f"{100*(sl<=b).mean():5.1f}%")

            # is the reference largely copied from the source? decides whether
            # a lead-k extractive baseline is a serious competitor
            import random
            random.seed(0)
            ov = []
            for d in random.sample(docs, min(200, len(docs))):
                s = set(tok.encode(d["summary"]).ids)
                t = set(tok.encode(d["document"]).ids)
                ov.append(len(s & t) / max(1, len(s)))
            print(f"  summary tokens also in document: {np.mean(ov):.3f} "
                  f"(1.0 = fully extractive vocabulary)")


if __name__ == "__main__":
    main()