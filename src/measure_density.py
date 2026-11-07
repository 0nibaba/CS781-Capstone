import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS = os.path.join(ROOT, "training_data/tokens")

BLOCK = 1024
VAL_TOKENS = 5_000_000
N_WINDOWS = 2000
SEED = 1899
EOS = 0

SOURCES = ["acts", "high", "supreme", "tribunal"]


def main():
    rng = np.random.default_rng(SEED)
    rows = {}

    for src in SOURCES:
        path = os.path.join(TOKENS, f"{src}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        train_end = len(arr) - VAL_TOKENS

        starts = rng.integers(0, train_end - BLOCK - 1, size=N_WINDOWS)
        seps = np.array([int((arr[s:s + BLOCK] == EOS).sum()) for s in starts])
        docs = seps + 1

        rows[src] = {
            "tokens": len(arr),
            "mean_docs": docs.mean(),
            "median_docs": float(np.median(docs)),
            "p90_docs": float(np.percentile(docs, 90)),
            "max_docs": int(docs.max()),
            "pct_single_doc": 100.0 * (seps == 0).mean(),
            "mean_frag_len": BLOCK / docs.mean(),
        }
        print(f"[{src}] done", flush=True)

    print("\n" + "=" * 88)
    print(f"{'source':<10} {'mean':>6} {'med':>5} {'p90':>5} {'max':>5} "
          f"{'% single-doc':>13} {'mean frag (tok)':>16}")
    print("-" * 88)
    for src in SOURCES:
        r = rows[src]
        print(f"{src:<10} {r['mean_docs']:>6.2f} {r['median_docs']:>5.0f} "
              f"{r['p90_docs']:>5.0f} {r['max_docs']:>5d} "
              f"{r['pct_single_doc']:>12.1f}% {r['mean_frag_len']:>16.0f}")
    print("=" * 88)
    print(f"{N_WINDOWS} random {BLOCK}-token windows per source, seed {SEED}")

    print("\ndensity ranking (most boundaries first):")
    for src in sorted(SOURCES, key=lambda s: -rows[s]["mean_docs"]):
        print(f"  {src:<10} {rows[src]['mean_docs']:.2f} docs/window")

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    out = os.path.join(ROOT, "logs", "separator_density.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()