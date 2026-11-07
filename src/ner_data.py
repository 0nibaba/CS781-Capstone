import os
import json
import random

import torch
from tokenizers import Tokenizer

from bio_tagging import spans_to_bio, O_ID
from windowing import make_windows, WINDOW, STRIDE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")
TEST_DIR = os.path.join(ROOT, "test_data/Phase_2_LNER/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

IGNORE = -100
PAD_ID = 0


def load_fold(fold):
    if fold == "fold3":
        with open(os.path.join(TEST_DIR, "fold3-text.json")) as f:
            return json.load(f), None
    with open(os.path.join(TRAIN_DIR, f"{fold}-text.json")) as f:
        docs = json.load(f)
    with open(os.path.join(TRAIN_DIR, f"{fold}-gold.json")) as f:
        gold = json.load(f)
    return docs, gold


class NERWindows:

    def __init__(self, docs, gold, tok, window=WINDOW, stride=STRIDE):
        self.items = []
        self.docs = docs
        self.offsets = {}

        for did, text in docs.items():
            enc = tok.encode(text)
            self.offsets[did] = enc.offsets
            ents = gold.get(did, []) if gold else []
            tags, _ = spans_to_bio(ents, enc.offsets, len(enc.ids))

            for s, e, kf, kt in make_windows(len(enc.ids), window, stride):
                self.items.append({
                    "doc_id": did,
                    "input_ids": enc.ids[s:e],
                    "labels": tags[s:e],
                    "tok_start": s,
                    "keep_from": kf - s,
                    "keep_to": kt - s,
                    "n_tokens": len(enc.ids),
                })

    def __len__(self):
        return len(self.items)


def collate(items, device):
    B = len(items)
    T = max(len(it["input_ids"]) for it in items)

    ids = torch.full((B, T), PAD_ID, dtype=torch.long)
    msk = torch.zeros((B, T), dtype=torch.long)
    lab = torch.full((B, T), IGNORE, dtype=torch.long)
    pos = torch.zeros((B, T), dtype=torch.float)

    for i, it in enumerate(items):
        n = len(it["input_ids"])
        ids[i, :n] = torch.tensor(it["input_ids"], dtype=torch.long)
        msk[i, :n] = 1
        lab[i, :n] = torch.tensor(it["labels"], dtype=torch.long)
        start, total = it["tok_start"], max(1, it["n_tokens"])
        pos[i, :n] = (torch.arange(n, dtype=torch.float) + start) / total

    return (ids.to(device, non_blocking=True),
            msk.to(device, non_blocking=True),
            lab.to(device, non_blocking=True),
            pos.to(device, non_blocking=True))


def batches(ds, batch_size, device, shuffle=True, seed=None):
    idx = list(range(len(ds)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [ds.items[j] for j in idx[i:i + batch_size]]
        yield chunk, collate(chunk, device)


def class_weights(ds, n_tags, device, scheme="inv_sqrt"):
    counts = torch.zeros(n_tags)
    for it in ds.items:
        for t in it["labels"]:
            counts[t] += 1
    counts = counts.clamp(min=1.0)

    if scheme == "none":
        w = torch.ones(n_tags)
    elif scheme == "inv":
        w = counts.sum() / counts
    elif scheme == "inv_sqrt":
        w = (counts.sum() / counts).sqrt()
    else:
        raise ValueError(scheme)

    w = w / w.mean()
    return w.to(device)


if __name__ == "__main__":
    tok = Tokenizer.from_file(TOKENIZER)
    d1, g1 = load_fold("fold1")
    ds = NERWindows(d1, g1, tok)
    print(f"fold1: {len(d1)} documents -> {len(ds)} windows")

    chunk, (ids, msk, lab) = next(iter(batches(ds, 8, "cpu", seed=0)))
    print(f"batch: ids {tuple(ids.shape)}  real tokens "
          f"{msk.sum().item()}/{msk.numel()}")
    sup = (lab != IGNORE).sum().item()
    ent = ((lab != IGNORE) & (lab != O_ID)).sum().item()
    print(f"labelled {sup}, of which entity tags {ent} "
          f"({100*ent/sup:.1f}%)")

    from bio_tagging import TAGS
    for scheme in ("none", "inv_sqrt", "inv"):
        w = class_weights(ds, len(TAGS), "cpu", scheme)
        print(f"\n  {scheme}: O weight {w[O_ID]:.3f}, "
              f"max {w.max():.1f} ({TAGS[w.argmax()]}), "
              f"ratio {w.max()/w[O_ID]:.0f}:1")