"""
What is in the Phase 3 Rhetorical Role data?

Three things decide the architecture:

1. DOCUMENT LENGTH in tokens. If a whole document fits in 1024, every
   sentence can be encoded with full document context in one pass. If not,
   the Phase 2 windowing machinery is needed again.

2. LABEL TRANSITIONS. Rhetorical roles follow a narrative order (Fact ->
   Issue -> Argument -> Ratio -> Ruling). If the transition matrix is sparse
   — most label pairs never adjacent — a sentence-level CRF should help as
   much as the BIO CRF did in Phase 2 (+0.18 macro-F1).

3. DOMAIN DIVERGENCE. If CL and IT have different label distributions, a
   joint model may be worse than two separate ones, or may need a domain
   feature.
"""

import os
import json
from collections import Counter, defaultdict

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_3_RR/Train")
TEST = os.path.join(ROOT, "test_data/Phase_3_RR/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")


def load(path):
    with open(path) as f:
        return json.load(f)

def load_docs(path):
    """Normalise both shapes into [{id, text, labels?}, ...].

    train/dev: list of {id, text, labels}
    test:      dict of {doc_id: [sentence, ...]}  — no labels
    """
    d = load(path)
    if isinstance(d, dict):
        return [{"id": k, "text": v} for k, v in d.items()]
    return d

def pct(a, q):
    return int(np.percentile(a, q))


def main():
    tok = Tokenizer.from_file(TOKENIZER)

    vocab_path = os.path.join(TRAIN, "label_vocab.json")
    vocab = load(vocab_path) if os.path.exists(vocab_path) else None
    print(f"label_vocab.json: {type(vocab).__name__}")
    print(f"  {vocab}\n")

    splits = {}
    for dom in ("CL", "IT"):
        for sp in ("train", "dev"):
            p = os.path.join(TRAIN, f"{dom}_{sp}.json")
            if os.path.exists(p):
                splits[f"{dom}_{sp}"] = load_docs(p)
        p = os.path.join(TEST, f"{dom}_test.json")
        if os.path.exists(p):
            splits[f"{dom}_test"] = load_docs(p)

    # ---- sizes ----
    print("=" * 74)
    print("splits")
    print("=" * 74)
    print(f"  {'split':<10} {'docs':>6} {'sentences':>10} {'sent/doc med':>13} "
          f"{'max':>6} {'labelled':>9}")
    for k, v in splits.items():
        n_sent = [len(d["text"]) for d in v]
        lab = sum(1 for d in v if d.get("labels"))
        print(f"  {k:<10} {len(v):>6} {sum(n_sent):>10,} "
              f"{int(np.median(n_sent)):>13} {max(n_sent):>6} {lab:>9}")
        # labels must align with sentences
        bad = [d["id"] for d in v
               if d.get("labels") and len(d["labels"]) != len(d["text"])]
        if bad:
            print(f"    WARNING: {len(bad)} documents where "
                  f"len(labels) != len(text)")

    # ---- token lengths ----
    print("\n" + "=" * 74)
    print("length in tokens (block_size is 1024)")
    print("=" * 74)
    for k, v in splits.items():
        sent_toks, doc_toks = [], []
        for d in v:
            lens = [len(tok.encode(s).ids) for s in d["text"]]
            sent_toks.extend(lens)
            doc_toks.append(sum(lens))
        s, t = np.array(sent_toks), np.array(doc_toks)
        print(f"  {k}")
        print(f"    sentence: median {pct(s,50):>4}  p90 {pct(s,90):>4}  "
              f"p99 {pct(s,99):>5}  max {s.max():>5}")
        print(f"    document: median {pct(t,50):>5,}  p90 {pct(t,90):>6,}  "
              f"max {t.max():>6,}   over 1024: "
              f"{100*(t > 1024).mean():.0f}%")

    # ---- label distribution ----
    print("\n" + "=" * 74)
    print("label distribution  (macro-F1 weights all labels equally)")
    print("=" * 74)
    counts = {}
    for k in ("CL_train", "IT_train", "CL_dev", "IT_dev"):
        if k in splits:
            counts[k] = Counter(l for d in splits[k] for l in d.get("labels", []))

    all_labels = sorted(set().union(*(c.keys() for c in counts.values())),
                        key=lambda l: -counts.get("CL_train", Counter())[l])
    cols = list(counts)
    print(f"  {'label':<24}" + "".join(f"{c:>12}" for c in cols)
          + f"{'CL%':>8}{'IT%':>8}")
    print("  " + "-" * (24 + 12 * len(cols) + 16))
    cl_t = sum(counts.get("CL_train", Counter()).values())
    it_t = sum(counts.get("IT_train", Counter()).values())
    for lab in all_labels:
        row = "".join(f"{counts[c].get(lab, 0):>12,}" for c in cols)
        clp = 100 * counts.get("CL_train", Counter()).get(lab, 0) / max(1, cl_t)
        itp = 100 * counts.get("IT_train", Counter()).get(lab, 0) / max(1, it_t)
        flag = "  <-- domain gap" if abs(clp - itp) > 5 else ""
        print(f"  {lab:<24}{row}{clp:>7.1f}%{itp:>7.1f}%{flag}")

    if vocab:
        declared = set(vocab if isinstance(vocab, list) else vocab.keys())
        seen = set(all_labels)
        if declared - seen:
            print(f"\n  declared but never seen in training: "
                  f"{sorted(declared - seen)}")
        if seen - declared:
            print(f"\n  seen but NOT in label_vocab: {sorted(seen - declared)}")

    rare = [l for l in all_labels
            if counts.get("CL_train", Counter())[l]
            + counts.get("IT_train", Counter())[l] < 100]
    if rare:
        print(f"\n  under 100 training examples total: {rare}")
        print("  these carry the same macro-F1 weight as the most frequent label")

    # ---- transition structure: the CRF case ----
    print("\n" + "=" * 74)
    print("label transitions — how constrained is the sequence?")
    print("=" * 74)
    for dom in ("CL", "IT"):
        k = f"{dom}_train"
        if k not in splits:
            continue
        trans = defaultdict(Counter)
        starts = Counter()
        n_pairs = 0
        for d in splits[k]:
            labs = d.get("labels", [])
            if labs:
                starts[labs[0]] += 1
            for a, b in zip(labs, labs[1:]):
                trans[a][b] += 1
                n_pairs += 1

        seen_pairs = sum(len(c) for c in trans.values())
        n_lab = len(all_labels)
        print(f"\n  {dom}: {seen_pairs} distinct transitions observed of "
              f"{n_lab*n_lab} possible ({100*seen_pairs/(n_lab*n_lab):.0f}%)")
        self_trans = sum(trans[l][l] for l in trans)
        print(f"     self-transitions (same label repeats): "
              f"{100*self_trans/max(1,n_pairs):.0f}% of all pairs")
        print(f"     documents start with: "
              + ", ".join(f"{l} ({n})" for l, n in starts.most_common(3)))
        print(f"     most common transitions:")
        pairs = [(n, a, b) for a in trans for b, n in trans[a].items()]
        for n, a, b in sorted(pairs, reverse=True)[:6]:
            print(f"       {a:<22} -> {b:<22} {n:>5}")

    print("\n  a sparse transition matrix and a high self-transition rate both")
    print("  favour a CRF over independent per-sentence classification")

    # ---- what a document looks like ----
    print("\n" + "=" * 74)
    print("example document")
    print("=" * 74)
    d = splits["CL_train"][0]
    print(f"  {d['id']}  ({len(d['text'])} sentences)")
    for s, l in list(zip(d["text"], d.get("labels", [])))[:10]:
        print(f"    {l:<22} {s[:88]}")


if __name__ == "__main__":
    main()