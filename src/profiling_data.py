import os
import json
import hashlib
import numpy as np
import pandas as pd
import ijson

RAW  = "/data1/yashy26/training_data/raw/Phase_1_Pretraining/Train"
TEST = "/data1/yashy26/test_data/Phase_1_Pretraining/Test/test.json"

FILES = {
    "acts":     f"{RAW}/train_acts.json",
    "high":     f"{RAW}/train_high_courts.json",
    "supreme":  f"{RAW}/train_supreme_court.json",
    "tribunal": f"{RAW}/train_tribunals.json",
}

WORDS_TO_TOKENS = 1.3
SHORT_DOC = 200


def stream_records(path):
    with open(path, "rb") as f:
        first = f.read(1)
        f.seek(0)
        if first == b"[":
            yield from ijson.items(f, "item")
        else:
            for _key, val in ijson.kvitems(f, ""):
                yield val


print("file sizes:")
for name, path in FILES.items():
    print(f"  {name:10s} {os.path.getsize(path) / 1e9:6.2f} GB")
print()

stats, hashes = {}, {}

for name, path in FILES.items():
    print(f"[{name}] streaming...", flush=True)

    wc_list, seen = [], set()
    dupes = header = short = n = 0

    for rec in stream_records(path):
        if not isinstance(rec, dict):
            continue
        text = rec.get("text")
        if not text:
            continue

        n += 1
        w = rec.get("word_count") or len(text.split())
        wc_list.append(w)
        if w < SHORT_DOC:
            short += 1

        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            dupes += 1
        seen.add(h)

        if text.lstrip().startswith("PETITIONER"):
            header += 1

        if n % 100_000 == 0:
            print(f"  ...{n:,} docs", flush=True)

    if n == 0:
        print(f"[{name}] WARNING: no usable records")
        continue

    wc = np.array(wc_list)
    hashes[name] = seen

    stats[name] = {
        "docs":         n,
        "dupes":        dupes,
        "words_total":  int(wc.sum()),
        "p5":           int(np.percentile(wc, 5)),
        "p50":          int(np.percentile(wc, 50)),
        "p95":          int(np.percentile(wc, 95)),
        "max":          int(wc.max()),
        "pct_short":    round(100 * short / n, 1),
        "pct_header":   round(100 * header / n, 1),
    }
    print(f"[{name}] {n:,} docs, {wc.sum():,} words", flush=True)
    del wc_list, wc

df = pd.DataFrame(stats).T
df["tokens_M"]  = (df["words_total"] * WORDS_TO_TOKENS / 1e6).round(1)
df["share_pct"] = (100 * df["words_total"] / df["words_total"].sum()).round(1)

print("\n" + "=" * 100)
print(df.to_string())
print("=" * 100)
print(f"TOTAL: {df['words_total'].sum():,} words "
      f"(~{df['tokens_M'].sum():.0f}M tokens at {WORDS_TO_TOKENS} tok/word)")

print("\ncross-source duplicates:")
names = list(hashes)
found = False
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        ov = len(hashes[a] & hashes[b])
        if ov:
            print(f"  {a} / {b}: {ov:,}")
            found = True
if not found:
    print("  none")

print("\ntest file:")
with open(TEST, encoding="utf-8") as f:
    test = json.load(f)
twc = np.array([len(v["text"].split()) for v in test.values()])
print(f"  {len(test)} prompts | words min={twc.min()} "
      f"median={np.median(twc):.0f} max={twc.max()}")

os.makedirs("/data1/yashy26/logs", exist_ok=True)
df.to_csv("/data1/yashy26/logs/corpus_profile.csv")
print("\nsaved to logs/corpus_profile.csv")