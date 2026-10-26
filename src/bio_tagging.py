"""
BIO tagging for L-NER, and its inverse.

The decode must invert the encode exactly, or the 98.18% alignment ceiling
measured in test_alignment.py is lost on top. The round-trip test at the
bottom checks precisely that: gold spans -> BIO tags -> decoded spans should
recover 8,240 / 8,393, no fewer.

Two decoding conventions for malformed sequences are compared, since models
will emit I-X without a preceding B-X:
  strict  - an orphan I-X is dropped
  lenient - an orphan I-X starts a new entity
"""

import os
import json
from collections import Counter, defaultdict

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
          "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]

TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}
O_ID = TAG2ID["O"]


# ---------- alignment (same convention as test_alignment.py) ----------

def char_span_to_tokens(offsets, start, end):
    """Token indices [i, j) whose character ranges overlap [start, end)."""
    idx = [k for k, (s, e) in enumerate(offsets)
           if s < end and e > start and e > s]
    return (idx[0], idx[-1] + 1) if idx else None


def tokens_to_char_span(offsets, i, j, text):
    """Token range -> character span, trimming whitespace. This is the 'trim'
    convention, measured at 98.18% exact recovery."""
    s, e = offsets[i][0], offsets[j - 1][1]
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


# ---------- encode ----------

def spans_to_bio(entities, offsets, n_tokens):
    """
    entities: list of [start, end, label, ...]
    Returns (tag_ids, n_dropped). An entity is dropped only if it maps to no
    token at all, or if its tokens are already claimed by an earlier entity.
    """
    tags = [O_ID] * n_tokens
    dropped = 0

    # longest first, so a short entity cannot steal tokens from a long one
    for ent in sorted(entities, key=lambda e: e[0] - e[1]):
        tr = char_span_to_tokens(offsets, ent[0], ent[1])
        if tr is None:
            dropped += 1
            continue
        i, j = tr
        if any(tags[k] != O_ID for k in range(i, j)):
            dropped += 1          # overlaps something already tagged
            continue
        tags[i] = TAG2ID[f"B-{ent[2]}"]
        for k in range(i + 1, j):
            tags[k] = TAG2ID[f"I-{ent[2]}"]
    return tags, dropped


# ---------- decode ----------

def bio_to_spans(tag_ids, offsets, text, lenient=True):
    """Returns [[start, end, label], ...] in document order."""
    out = []
    i = 0
    n = len(tag_ids)
    while i < n:
        tag = ID2TAG[tag_ids[i]]
        if tag == "O":
            i += 1
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and not lenient:
            i += 1
            continue
        j = i + 1
        while j < n and ID2TAG[tag_ids[j]] == f"I-{label}":
            j += 1
        s, e = tokens_to_char_span(offsets, i, j, text)
        if e > s:
            out.append([s, e, label])
        i = j
    return out


# ---------- tests ----------

def load(name):
    with open(os.path.join(TRAIN, name)) as f:
        return json.load(f)


def main():
    tok = Tokenizer.from_file(TOKENIZER)
    docs, gold = {}, {}
    for fold in ("fold1", "fold2"):
        docs.update(load(f"{fold}-text.json"))
        gold.update(load(f"{fold}-gold.json"))

    print(f"{len(TAGS)} tags: O + {len(LABELS)} x (B, I)\n")
    print("tokenizing...", flush=True)

    total = sum(len(v) for v in gold.values())
    dropped_enc = 0
    tag_counts = Counter()
    recovered = {"strict": 0, "lenient": 0}
    by_label = {m: defaultdict(lambda: [0, 0]) for m in recovered}
    adjacent_same = 0
    misses = []

    for did, ents in gold.items():
        text = docs[did]
        enc = tok.encode(text)
        offsets = enc.offsets

        tags, d = spans_to_bio(ents, offsets, len(enc.ids))
        dropped_enc += d
        tag_counts.update(ID2TAG[t] for t in tags)

        # entities of the same label sharing a token boundary would merge
        order = sorted([e for e in ents], key=lambda e: e[0])
        for a, b in zip(order, order[1:]):
            if a[2] == b[2]:
                ta = char_span_to_tokens(offsets, a[0], a[1])
                tb = char_span_to_tokens(offsets, b[0], b[1])
                if ta and tb and ta[1] == tb[0]:
                    adjacent_same += 1

        gold_set = {(e[0], e[1], e[2]) for e in ents}
        for mode in ("strict", "lenient"):
            got = bio_to_spans(tags, offsets, text, lenient=(mode == "lenient"))
            got_set = {tuple(g) for g in got}
            for e in ents:
                key = (e[0], e[1], e[2])
                by_label[mode][e[2]][1] += 1
                if key in got_set:
                    recovered[mode] += 1
                    by_label[mode][e[2]][0] += 1
                elif mode == "lenient" and len(misses) < 10:
                    near = [g for g in got if abs(g[0] - e[0]) < 30]
                    misses.append((e[2], e[0], e[1], text[e[0]:e[1]],
                                   near[:2]))

    print(f"\n{'=' * 72}")
    print(f"round trip: gold spans -> BIO -> spans   ({total:,} entities)")
    print(f"{'=' * 72}")
    print(f"  dropped at encode (no tokens, or token overlap): {dropped_enc}")
    print(f"  same-label entities sharing a token boundary: {adjacent_same}")
    print()
    for mode in ("strict", "lenient"):
        print(f"  {mode:<8} {recovered[mode]:>6,} / {total:,} = "
              f"{100*recovered[mode]/total:6.2f}%")
    print(f"\n  alignment ceiling was 8,240 / 8,393 = 98.18%")
    print(f"  BIO must not lose anything beyond that")

    m = "lenient"
    macro = sum(ok / n for ok, n in by_label[m].values()) / len(by_label[m])
    print(f"\n  macro average ({m}): {100*macro:.2f}%  "
          f"(alignment macro ceiling 98.66%)")

    print(f"\n{'=' * 72}")
    print("tag distribution — this is the class imbalance")
    print(f"{'=' * 72}")
    tot_tags = sum(tag_counts.values())
    print(f"  {'O':<14} {tag_counts['O']:>9,}  {100*tag_counts['O']/tot_tags:6.2f}%")
    ent_tags = tot_tags - tag_counts["O"]
    print(f"  {'(entity tags)':<14} {ent_tags:>9,}  "
          f"{100*ent_tags/tot_tags:6.2f}%")
    print(f"\n  O outnumbers entity tokens {tag_counts['O']/ent_tags:.1f}:1")
    print("  rarest tags:")
    for t, c in sorted(tag_counts.items(), key=lambda kv: kv[1])[:6]:
        print(f"    {t:<14} {c:>7,}  {100*c/tot_tags:.4f}%")

    if misses:
        print(f"\n{'=' * 72}")
        print("entities not recovered")
        print(f"{'=' * 72}")
        for lab, s, e, txt, near in misses:
            print(f"  {lab:<10} gold [{s},{e}] {txt!r}")
            for g in near:
                print(f"             near [{g[0]},{g[1]}] {g[2]}")


if __name__ == "__main__":
    main()