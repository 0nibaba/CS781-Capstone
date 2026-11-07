import json
import random
import argparse
from collections import defaultdict
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_eval_set(pool_path, out_path, num_samples=200, seed=42,
                   prompt_words=512, ref_words=128, max_windows_per_doc=3):
    random.seed(seed)

    with open(pool_path) as f:
        pool = json.load(f)

    need = prompt_words + ref_words

    by_source = defaultdict(list)
    for source, texts in pool.items():
        for text in texts:
            words = text.split()
            if len(words) >= need:
                by_source[source].append(words)

    print("eligible documents per source:")
    for src in sorted(by_source):
        print(f"  {src:10s} {len(by_source[src]):5d}")
    print()

    sources = sorted(by_source)
    per_source = num_samples // len(sources)

    eval_set = {}
    n = 1

    for source in sources:
        docs = by_source[source]
        want = per_source
        made = 0

        for rounds in range(max_windows_per_doc):
            if made >= want:
                break
            remaining = want - made
            picks = random.sample(docs, min(remaining, len(docs)))

            for words in picks:
                start = random.randint(0, len(words) - need)
                eval_set[f"eval_{n:04d}"] = {
                    "prompt":    " ".join(words[start:start + prompt_words]),
                    "reference": " ".join(words[start + prompt_words:start + need]),
                    "source":    source,
                }
                n += 1
                made += 1
                if made >= want:
                    break

        if made < want:
            print(f"  WARNING: {source} only produced {made}/{want} samples")

    with open(out_path, "w") as f:
        json.dump(eval_set, f, indent=2)

    counts = defaultdict(int)
    for v in eval_set.values():
        counts[v["source"]] += 1
    print(f"built {len(eval_set)} samples -> {out_path}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    k = list(eval_set)[0]
    ex = eval_set[k]
    print(f"\nsample {k} ({ex['source']}):")
    print(f"  prompt words:    {len(ex['prompt'].split())}")
    print(f"  reference words: {len(ex['reference'].split())}")
    print(f"  prompt ends:     ...{ex['prompt'][-90:]!r}")
    print(f"  reference begins: {ex['reference'][:90]!r}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool_path", default=os.path.join(ROOT, "test_data/heldout_pool.json"))
    p.add_argument("--out_path",  default=os.path.join(ROOT, "test_data/eval_set.json"))
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_eval_set(args.pool_path, args.out_path, args.num_samples, args.seed)