"""
What is in the Phase 2 L-NER data?

Macro-F1 over 12 labels means rare entity types count as much as common ones,
so the label distribution matters more than it would under micro-F1. Document
length matters because the Phase 1 models have a fixed 1024-token block and
legal judgments are long.
"""

import os
import json
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")
TEST = os.path.join(ROOT, "test_data/Phase_2_LNER/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")


def load(path):
    with open(path) as f:
        return json.load(f)


def pct(a, q):
    return int(np.percentile(a, q))


def main():
    tok = Tokenizer.from_file(TOKENIZER)

    with open(os.path.join(TRAIN, "labels.txt")) as f:
        labels = [l.strip() for l in f if l.strip()]
    print(f"labels ({len(labels)}): {labels}\n")

    docs, gold = {}, {}
    for fold in ("fold1", "fold2"):
        t = load(os.path.join(TRAIN, f"{fold}-text.json"))
        g = load(os.path.join(TRAIN, f"{fold}-gold.json"))
        print(f"{fold}: {len(t)} documents, {len(g)} with annotations")
        docs.update(t)
        gold.update(g)

    test = load(os.path.join(TEST, "fold3-text.json"))
    print(f"fold3 (test): {len(test)} documents\n")

    # ---- document lengths ----
    print("=" * 70)
    print("document length")
    print("=" * 70)
    for name, d in (("train", docs), ("test", test)):
        chars = np.array([len(v) for v in d.values()])
        toks = np.array([len(tok.encode(v).ids) for v in d.values()])
        print(f"  {name}: {len(d)} docs")
        print(f"    chars  median {pct(chars,50):,}  p90 {pct(chars,90):,}  "
              f"max {chars.max():,}")
        print(f"    tokens median {pct(toks,50):,}  p90 {pct(toks,90):,}  "
              f"max {toks.max():,}")
        over = (toks > 1024).sum()
        print(f"    exceed 1024 tokens: {over} ({100*over/len(toks):.1f}%)")

    # ---- entities ----
    print("\n" + "=" * 70)
    print("entities")
    print("=" * 70)
    counts = Counter()
    span_chars, span_toks, per_doc = [], [], []
    for did, ents in gold.items():
        per_doc.append(len(ents))
        text = docs.get(did, "")
        for e in ents:
            s, en, lab = e[0], e[1], e[2]
            counts[lab] += 1
            span_chars.append(en - s)
            if text:
                span_toks.append(len(tok.encode(text[s:en]).ids))

    total = sum(counts.values())
    print(f"  {total:,} entities across {len(gold)} documents")
    print(f"  per document: median {pct(per_doc,50)}  p90 {pct(per_doc,90)}  "
          f"max {max(per_doc)}\n")

    print(f"  {'label':<12} {'count':>8} {'share':>8}  (macro-F1 weights these equally)")
    print("  " + "-" * 44)
    for lab in sorted(counts, key=counts.get, reverse=True):
        print(f"  {lab:<12} {counts[lab]:>8,} {100*counts[lab]/total:>7.2f}%")
    missing = set(labels) - set(counts)
    if missing:
        print(f"\n  labels with NO training examples: {sorted(missing)}")
    rare = [l for l in counts if counts[l] < 100]
    if rare:
        print(f"  labels with <100 examples: {rare}")

    sc = np.array(span_chars)
    print(f"\n  span length (chars):  median {pct(sc,50)}  p90 {pct(sc,90)}  "
          f"max {sc.max()}")
    if span_toks:
        st = np.array(span_toks)
        print(f"  span length (tokens): median {pct(st,50)}  p90 {pct(st,90)}  "
              f"max {st.max()}")

    # ---- sanity checks on the annotations themselves ----
    print("\n" + "=" * 70)
    print("annotation sanity")
    print("=" * 70)
    bad_range, text_mismatch, overlapping, with_text = 0, 0, 0, 0
    for did, ents in gold.items():
        text = docs.get(did)
        if text is None:
            continue
        spans = sorted((e[0], e[1]) for e in ents)
        for i in range(1, len(spans)):
            if spans[i][0] < spans[i-1][1]:
                overlapping += 1
        for e in ents:
            s, en = e[0], e[1]
            if not (0 <= s < en <= len(text)):
                bad_range += 1
                continue
            if len(e) > 3 and e[3]:
                with_text += 1
                if text[s:en] != e[3]:
                    text_mismatch += 1

    print(f"  out-of-range spans: {bad_range}")
    print(f"  overlapping spans: {overlapping}")
    if with_text:
        print(f"  entity_text supplied: {with_text:,}, "
              f"mismatching text[start:end]: {text_mismatch}")
        print("    (a mismatch would mean the offsets are not plain Python "
              "character indices)")
    else:
        print("  no entity_text field to cross-check offsets against")

    # ---- what an entity actually looks like ----
    print("\n" + "=" * 70)
    print("examples")
    print("=" * 70)
    did = next(iter(gold))
    text = docs[did]
    print(f"  document {did}, {len(text):,} chars, {len(gold[did])} entities")
    for e in gold[did][:8]:
        s, en, lab = e[0], e[1], e[2]
        print(f"    [{s:6d},{en:6d}] {lab:<10} {text[s:en]!r}")
        print(f"      context: ...{text[max(0,s-25):s]!r} >>> "
              f"{text[en:en+25]!r}...")


if __name__ == "__main__":
    main()