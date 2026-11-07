import os
import json

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training_data/raw/Phase_3_RR/Train")
TEST_DIR = os.path.join(ROOT, "test_data/Phase_3_RR/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

IGNORE = -100
PAD_ID = 0

MAX_LEN = 1024
CONTEXT_SENTS = 8
MAX_SENT_TOKENS = 256

with open(os.path.join(TRAIN_DIR, "label_vocab.json")) as f:
    LABEL2ID = json.load(f)
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABELS = [ID2LABEL[i] for i in range(len(ID2LABEL))]
N_LABELS = len(LABELS)

SEP_TOKEN = "<|unused0|>"
DOMAIN_TOKEN = {"CL": "<|unused1|>", "IT": "<|unused2|>"}


def load_docs(path, domain):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        return [{"id": k, "domain": domain, "text": v} for k, v in d.items()]
    return [{**x, "domain": domain} for x in d]


def load_split(split, domains=("CL", "IT")):
    out = []
    for dom in domains:
        if split == "test":
            p = os.path.join(TEST_DIR, f"{dom}_test.json")
        else:
            p = os.path.join(TRAIN_DIR, f"{dom}_{split}.json")
        if os.path.exists(p):
            out.extend(load_docs(p, dom))
    return out


class RRBlocks:

    def __init__(self, docs, tok, max_len=MAX_LEN, context=CONTEXT_SENTS,
                 max_sent=MAX_SENT_TOKENS, ctx_frac=0.34):
        self.items = []
        self.docs = {d["id"]: d for d in docs}

        sep = tok.token_to_id(SEP_TOKEN)
        assert sep is not None, f"{SEP_TOKEN} missing from the tokenizer"

        for d in docs:
            dom_id = tok.token_to_id(DOMAIN_TOKEN[d["domain"]])
            sent_ids = [e.ids[:max_sent] for e in tok.encode_batch(d["text"])]
            labels = [LABEL2ID[l] for l in d["labels"]] if d.get("labels") \
                else [IGNORE] * len(sent_ids)
            n = len(sent_ids)

            start = 0
            while start < n:
                budget = max_len - 1

                ctx_from, ctx_cost, cap = start, 0, int(budget * ctx_frac)
                while ctx_from > 0 and start - ctx_from < context:
                    need = len(sent_ids[ctx_from - 1]) + 1
                    if ctx_cost + need > cap:
                        break
                    ctx_cost += need
                    ctx_from -= 1
                budget -= ctx_cost

                end = start
                while end < n:
                    need = len(sent_ids[end]) + 1
                    if need > budget and end > start:
                        break
                    budget -= need
                    end += 1
                    if budget <= 0:
                        break
                assert end > start, "no room to predict even one sentence"

                ids = [dom_id]
                markers = []
                for i in range(ctx_from, end):
                    ids.extend(sent_ids[i])
                    ids.append(sep)
                    if i >= start:
                        markers.append((i, len(ids) - 1))
                assert len(markers) == end - start

                self.items.append({
                    "doc_id": d["id"],
                    "domain": d["domain"],
                    "input_ids": ids,
                    "markers": markers,
                    "pred_from": start,
                    "pred_to": end,
                    "labels": labels,
                    "n_sents": n,
                })
                start = end

    def __len__(self):
        return len(self.items)


def collate(items, device):
    B = len(items)
    T = max(len(it["input_ids"]) for it in items)
    S = max(it["pred_to"] - it["pred_from"] for it in items)

    ids = torch.full((B, T), PAD_ID, dtype=torch.long)
    msk = torch.zeros((B, T), dtype=torch.long)
    pos = torch.zeros((B, S), dtype=torch.long)
    pmask = torch.zeros((B, S), dtype=torch.long)
    lab = torch.full((B, S), IGNORE, dtype=torch.long)

    for b, it in enumerate(items):
        k = len(it["input_ids"])
        ids[b, :k] = torch.tensor(it["input_ids"], dtype=torch.long)
        msk[b, :k] = 1
        keep = [(si, p) for si, p in it["markers"]
                if it["pred_from"] <= si < it["pred_to"]]
        for j, (si, p) in enumerate(keep):
            pos[b, j] = p
            pmask[b, j] = 1
            lab[b, j] = it["labels"][si]

    to = lambda t: t.to(device, non_blocking=True)
    return to(ids), to(msk), to(pos), to(pmask), to(lab)


def batches(ds, batch_size, device, shuffle=True, seed=None):
    import random
    idx = list(range(len(ds)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [ds.items[j] for j in idx[i:i + batch_size]]
        yield chunk, collate(chunk, device)


if __name__ == "__main__":
    tok = Tokenizer.from_file(TOKENIZER)

    for t in [SEP_TOKEN] + list(DOMAIN_TOKEN.values()):
        assert tok.token_to_id(t) is not None, f"{t} missing"
    print(f"{N_LABELS} labels: {LABELS}")
    print(f"markers: sep={tok.token_to_id(SEP_TOKEN)}, "
          f"CL={tok.token_to_id(DOMAIN_TOKEN['CL'])}, "
          f"IT={tok.token_to_id(DOMAIN_TOKEN['IT'])}\n")

    docs = load_split("train")
    ds = RRBlocks(docs, tok)
    print(f"{len(docs)} documents -> {len(ds)} blocks "
          f"({len(ds)/len(docs):.1f} per document)")

    n_sent = sum(len(d["text"]) for d in docs)
    pred = sum(it["pred_to"] - it["pred_from"] for it in ds.items)
    print(f"{n_sent:,} sentences, {pred:,} predicted slots")

    print("\n--- 1. every sentence predicted exactly once ---")
    from collections import Counter
    cov = Counter()
    for it in ds.items:
        for si in range(it["pred_from"], it["pred_to"]):
            cov[(it["doc_id"], si)] += 1
    expected = {(d["id"], i) for d in docs for i in range(len(d["text"]))}
    missing = expected - set(cov)
    dupes = {k for k, v in cov.items() if v > 1}
    print(f"  missing {len(missing)}, duplicated {len(dupes)}")
    assert not missing and not dupes

    print("\n--- 2. blocks fit, and markers land on the separator ---")
    sep = tok.token_to_id(SEP_TOKEN)
    too_long = [it for it in ds.items if len(it["input_ids"]) > MAX_LEN]
    print(f"  blocks over {MAX_LEN} tokens: {len(too_long)}")
    assert not too_long
    bad = 0
    for it in ds.items:
        for si, p in it["markers"]:
            if it["input_ids"][p] != sep:
                bad += 1
    print(f"  marker positions not on a separator: {bad}")
    assert bad == 0

    lens = sorted(len(it["input_ids"]) for it in ds.items)
    n_pred = sorted(it["pred_to"] - it["pred_from"] for it in ds.items)
    print(f"  block tokens: median {lens[len(lens)//2]}, max {lens[-1]}")
    print(f"  sentences predicted per block: median "
          f"{n_pred[len(n_pred)//2]}, max {n_pred[-1]}")

    print("\n--- 3. labels round-trip through blocks ---")
    rec = {d["id"]: [None] * len(d["text"]) for d in docs}
    for it in ds.items:
        for si in range(it["pred_from"], it["pred_to"]):
            rec[it["doc_id"]][si] = ID2LABEL[it["labels"][si]]
    bad = sum(1 for d in docs if rec[d["id"]] != d["labels"])
    print(f"  documents whose labels do not reconstruct: {bad}")
    assert bad == 0

    print("\n--- 4. a collated batch ---")
    chunk, (ids, msk, pos, pmask, lab) = next(iter(batches(ds, 4, "cpu", seed=0)))
    print(f"  ids {tuple(ids.shape)}  real tokens "
          f"{100*msk.float().mean():.0f}%")
    print(f"  markers {tuple(pos.shape)}  valid slots {int(pmask.sum())}")
    print(f"  labels: {int((lab != IGNORE).sum())} supervised, "
          f"all within range: "
          f"{bool(((lab == IGNORE) | ((lab >= 0) & (lab < N_LABELS))).all())}")
    g = ids.gather(1, pos)
    ok = ((g == sep) | (pmask == 0)).all()
    print(f"  every valid marker gathers a separator token: {bool(ok)}")
    assert ok