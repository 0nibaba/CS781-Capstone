import os
import json
from collections import Counter

from tokenizers import Tokenizer
from bio_tagging import (LABELS, TAGS, TAG2ID, ID2TAG, O_ID,
                         spans_to_bio, bio_to_spans, char_span_to_tokens,
                         tokens_to_char_span)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

WINDOW = 1024
STRIDE = 256
MARGIN = (WINDOW - STRIDE) // 2


def make_windows(n_tokens, window=WINDOW, stride=STRIDE):
    if n_tokens <= window:
        yield (0, n_tokens, 0, n_tokens)
        return

    starts = list(range(0, n_tokens - window + 1, stride))
    if starts[-1] + window < n_tokens:
        starts.append(n_tokens - window)

    cursor = 0
    for k, s in enumerate(starts):
        e = min(s + window, n_tokens)
        keep_to = n_tokens if k == len(starts) - 1 else e - MARGIN
        if keep_to > cursor:
            yield (s, e, cursor, keep_to)
            cursor = keep_to


def encode_document(text, entities, tok, window=WINDOW, stride=STRIDE):
    enc = tok.encode(text)
    offsets, ids = enc.offsets, enc.ids
    tags, _ = spans_to_bio(entities, offsets, len(ids)) if entities is not None \
        else ([O_ID] * len(ids), 0)

    out = []
    for s, e, kf, kt in make_windows(len(ids), window, stride):
        out.append({
            "input_ids": ids[s:e],
            "tags": tags[s:e],
            "tok_start": s,
            "keep_from": kf - s,
            "keep_to": kt - s,
        })
    return out, offsets, len(ids)


def merge_windows(windows, n_tokens, pred_key="tags"):
    full = [O_ID] * n_tokens
    for w in windows:
        s = w["tok_start"]
        for i in range(w["keep_from"], w["keep_to"]):
            full[s + i] = w[pred_key][i]
    return full


def load(name):
    with open(os.path.join(TRAIN, name)) as f:
        return json.load(f)


def main():
    tok = Tokenizer.from_file(TOKENIZER)
    docs, gold = {}, {}
    for fold in ("fold1", "fold2"):
        docs.update(load(f"{fold}-text.json"))
        gold.update(load(f"{fold}-gold.json"))

    print(f"window {WINDOW}, stride {STRIDE}, margin {MARGIN}\n")

    print("--- 1. coverage: every token predicted exactly once ---")
    for n in (500, 1024, 1025, 3000, 5542, 23781):
        cov = Counter()
        wins = list(make_windows(n))
        for s, e, kf, kt in wins:
            for i in range(kf, kt):
                cov[i] += 1
        bad = [i for i in range(n) if cov[i] != 1]
        print(f"  {n:>6} tokens -> {len(wins):>3} windows, "
              f"{'OK' if not bad else f'BAD: {len(bad)} tokens covered {set(cov[i] for i in bad)}x'}")
        assert not bad
    print()

    print("--- 2. round trip through windowing ---", flush=True)
    total = recovered = 0
    n_windows = 0
    straddle = 0
    misses = []

    for did, ents in gold.items():
        text = docs[did]
        wins, offsets, n_tok = encode_document(text, ents, tok)
        n_windows += len(wins)

        for e in ents:
            tr = char_span_to_tokens(offsets, e[0], e[1])
            if tr is None:
                continue
            i, j = tr
            owner = [w for w in wins
                     if w["tok_start"] + w["keep_from"] <= i
                     < w["tok_start"] + w["keep_to"]]
            if owner:
                w = owner[0]
                if j > w["tok_start"] + w["keep_to"]:
                    straddle += 1

        merged = merge_windows(wins, n_tok)
        got = {tuple(g) for g in bio_to_spans(merged, offsets, text)}

        for e in ents:
            total += 1
            if (e[0], e[1], e[2]) in got:
                recovered += 1
            elif len(misses) < 8:
                misses.append((e[2], e[0], e[1], text[e[0]:e[1]]))

    print(f"  {n_windows:,} windows over {len(gold)} documents "
          f"({n_windows/len(gold):.1f} per document)")
    print(f"  entities whose span crosses out of its owner's keep region: "
          f"{straddle}")
    print(f"\n  recovered {recovered:,} / {total:,} = "
          f"{100*recovered/total:.2f}%")
    print(f"  without windowing this was 98.18% — windowing must not lose more")

    if misses:
        print("\n  not recovered:")
        for lab, s, e, txt in misses:
            print(f"    {lab:<10} [{s},{e}] {txt!r}")

    print("\n--- 3. cost ---")
    tokens_raw = sum(len(tok.encode(docs[d]).ids) for d in gold)
    tokens_win = n_windows * WINDOW
    print(f"  document tokens {tokens_raw:,}")
    print(f"  tokens processed {tokens_win:,}  ({tokens_win/tokens_raw:.2f}x)")


if __name__ == "__main__":
    main()