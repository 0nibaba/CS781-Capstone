"""
Build a held-out pool, strictly disjoint from the tokenizer training sample.
Serves two purposes:
  - eval_sample.json  : per-source text for fertility measurement
  - heldout_pool.json : longer docs, for the 512/128 BERTScore eval set later
"""

import os
import json
import hashlib
import random
import ijson

RAW = "/data1/yashy26/training_data/raw/Phase_1_Pretraining/Train"
OUT = "/data1/yashy26/test_data"

FILES = {
    "acts":     f"{RAW}/train_acts.json",
    "high":     f"{RAW}/train_high_courts.json",
    "supreme":  f"{RAW}/train_supreme_court.json",
    "tribunal": f"{RAW}/train_tribunals.json",
}

# must exceed whatever your tokenizer sampler consumed from each file
SKIP_BYTES = {"acts": 150_000_000, "supreme": 200_000_000,
              "high": 80_000_000,  "tribunal": 80_000_000}

N_EVAL     = 300      # docs per source for fertility
N_POOL     = 500      # docs per source for the generation eval
MIN_WORDS  = 300      # fertility sample: avoid stubs
POOL_WORDS = 700      # generation eval needs 512 + 128 + slack
SEED = 1234


def stream(path):
    with open(path, "rb") as f:
        first = f.read(1)
        f.seek(0)
        if first == b"[":
            yield from ijson.items(f, "item")
        else:
            for _k, v in ijson.kvitems(f, ""):
                yield v


random.seed(SEED)
eval_sample, heldout_pool, manifest = {}, {}, {}

for name, path in FILES.items():
    print(f"[{name}] scanning past first {SKIP_BYTES[name]/1e6:.0f} MB ...", flush=True)

    consumed = 0
    cands_eval, cands_pool = [], []

    for rec in stream(path):
        if not isinstance(rec, dict):
            continue
        text = rec.get("text")
        if not text:
            continue

        consumed += len(text.encode("utf-8"))
        if consumed < SKIP_BYTES[name]:
            continue                       # region the tokenizer may have seen

        w = len(text.split())
        if w >= POOL_WORDS and len(cands_pool) < N_POOL * 4:
            cands_pool.append(text)
        elif w >= MIN_WORDS and len(cands_eval) < N_EVAL * 4:
            cands_eval.append(text)

        if len(cands_pool) >= N_POOL * 4 and len(cands_eval) >= N_EVAL * 4:
            break

    ev = random.sample(cands_eval, min(N_EVAL, len(cands_eval)))
    po = random.sample(cands_pool, min(N_POOL, len(cands_pool)))

    eval_sample[name] = ev
    heldout_pool[name] = po
    manifest[name] = sorted(hashlib.md5(t.encode()).hexdigest() for t in ev + po)

    print(f"[{name}] eval={len(ev)} pool={len(po)}", flush=True)

os.makedirs(OUT, exist_ok=True)
json.dump(eval_sample,  open(f"{OUT}/eval_sample.json", "w"))
json.dump(heldout_pool, open(f"{OUT}/heldout_pool.json", "w"))
json.dump(manifest,     open(f"{OUT}/heldout_hashes.json", "w"))

print("\nwrote eval_sample.json, heldout_pool.json, heldout_hashes.json")
print("IMPORTANT: exclude heldout_hashes.json from the training memmaps")