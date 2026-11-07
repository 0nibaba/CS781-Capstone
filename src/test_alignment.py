import os
import json
from collections import Counter, defaultdict

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

TRAILING_PUNCT = ".,;:)]}"


def load(name):
    with open(os.path.join(TRAIN, name)) as f:
        return json.load(f)


def char_span_to_tokens(offsets, start, end):
    idx = [k for k, (s, e) in enumerate(offsets)
           if s < end and e > start and e > s]
    return (idx[0], idx[-1] + 1) if idx else None


def tokens_to_char_span(offsets, i, j, text, mode):
    s, e = offsets[i][0], offsets[j - 1][1]
    if mode in ("trim", "trim_punct"):
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
    if mode == "trim_punct":
        while e > s and text[e - 1] in TRAILING_PUNCT:
            e -= 1
        while e > s and text[e - 1].isspace():
            e -= 1
    return s, e

def main():
    tok = Tokenizer.from_file(TOKENIZER)

    docs, gold = {}, {}
    for fold in ("fold1", "fold2"):
        docs.update(load(f"{fold}-text.json"))
        gold.update(load(f"{fold}-gold.json"))

    modes = ["strict", "expand", "trim", "trim_punct"]
    exact = {m: 0 for m in modes}
    by_label = {m: defaultdict(lambda: [0, 0]) for m in modes}
    failures = {m: [] for m in modes}
    unmappable = 0
    total = 0

    ends_punct = Counter()
    for did, ents in gold.items():
        for e in ents:
            if e[1] > e[0] and docs[did][e[1] - 1] in TRAILING_PUNCT:
                ends_punct[e[2]] += 1
    if ends_punct:
        print("gold spans that legitimately END in punctuation (trim_punct would break these):")
        for lab, n in ends_punct.most_common():
            print(f"  {lab:<12} {n}")
        print(f"  total {sum(ends_punct.values())}\n")
    else:
        print("no gold span ends in punctuation — stripping it is safe\n")

    print("tokenizing 70 documents...", flush=True)
    for did, ents in gold.items():
        text = docs[did]
        enc = tok.encode(text)
        offsets = enc.offsets

        for e in ents:
            gs, ge, lab = e[0], e[1], e[2]
            total += 1

            tr = char_span_to_tokens(offsets, gs, ge)
            if tr is None:
                unmappable += 1
                continue
            i, j = tr

            for m in modes:
                if m == "strict":
                    ok = (offsets[i][0] == gs and offsets[j - 1][1] == ge)
                    ps, pe = offsets[i][0], offsets[j - 1][1]
                else:
                    ps, pe = tokens_to_char_span(offsets, i, j, text, m)
                    ok = (ps == gs and pe == ge)

                by_label[m][lab][1] += 1
                if ok:
                    exact[m] += 1
                    by_label[m][lab][0] += 1
                elif len(failures[m]) < 12:
                    failures[m].append((lab, gs, ge, ps, pe,
                                        text[gs:ge], text[ps:pe]))

    print(f"\n{'=' * 72}")
    print(f"round-trip exactness over {total:,} gold spans")
    print(f"{'=' * 72}")
    if unmappable:
        print(f"  {unmappable} spans mapped to NO tokens at all\n")

    for m in modes:
        print(f"  {m:<8} {exact[m]:>6,} / {total:,}  "
              f"= {100*exact[m]/total:6.2f}%   <- ceiling on recall")

    best = max(modes, key=lambda m: exact[m])
    print(f"\n  best convention: {best}")

    print(f"\n{'=' * 72}")
    print(f"per-label exactness under '{best}'  (macro-F1 weights these equally)")
    print(f"{'=' * 72}")
    print(f"  {'label':<12} {'ok':>6} {'total':>7} {'%':>8}")
    print("  " + "-" * 36)
    rows = sorted(by_label[best].items(), key=lambda kv: kv[1][0] / kv[1][1])
    for lab, (ok, n) in rows:
        print(f"  {lab:<12} {ok:>6} {n:>7} {100*ok/n:>7.2f}%")
    macro = sum(ok / n for ok, n in by_label[best].values()) / len(by_label[best])
    print(f"\n  macro average: {100*macro:.2f}%  <- ceiling on macro-F1")

    print(f"\n{'=' * 72}")
    print(f"example failures under '{best}'")
    print(f"{'=' * 72}")
    for lab, gs, ge, ps, pe, gold_t, pred_t in failures[best]:
        print(f"  {lab:<10} gold [{gs},{ge}] {gold_t!r}")
        print(f"             got  [{ps},{pe}] {pred_t!r}")

    print(f"\n{'=' * 72}")
    print("failure shapes")
    print(f"{'=' * 72}")
    shapes = Counter()
    for lab, gs, ge, ps, pe, _, _ in failures[best]:
        shapes[("start " + ("early" if ps < gs else "late" if ps > gs else "ok"),
                "end " + ("early" if pe < ge else "late" if pe > ge else "ok"))] += 1
    for k, v in shapes.most_common():
        print(f"  {k[0]:<12} {k[1]:<10} {v}")


if __name__ == "__main__":
    main()