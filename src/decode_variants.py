import os
import json
import argparse

from tokenizers import Tokenizer

from bio_tagging import ID2TAG, O_ID, tokens_to_char_span
from ner_data import load_fold, TOKENIZER
from ner_metrics import strict_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def decode(tags, offsets, text, conf=None, lenient=True, bridge=0,
           min_chars=0, min_conf=0.0):
    out = []
    n = len(tags)
    i = 0
    while i < n:
        tag = ID2TAG[tags[i]]
        if tag == "O":
            i += 1
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and not lenient:
            i += 1
            continue

        j = i + 1
        last_good = j
        gap = 0
        while j < n:
            t = ID2TAG[tags[j]]
            if t == f"I-{label}":
                j += 1
                last_good = j
                gap = 0
            elif gap < bridge and t != f"B-{label}":
                j += 1
                gap += 1
            else:
                break
        j = last_good

        s, e = tokens_to_char_span(offsets, i, j, text)
        keep = e > s and (e - s) >= min_chars
        if keep and conf is not None and min_conf > 0:
            keep = sum(conf[i:j]) / max(1, j - i) >= min_conf
        if keep:
            out.append([s, e, label])
        i = max(j, i + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="tags_<fold>.json")
    ap.add_argument("--fold", required=True, choices=["fold1", "fold2"])
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOKENIZER)
    docs, gold = load_fold(args.fold)
    gold = {k: [[e[0], e[1], e[2]] for e in v] for k, v in gold.items()}

    saved = json.load(open(args.tags))
    offsets = {d: tok.encode(t).offsets for d, t in docs.items()}

    variants = [
        ("baseline (lenient)",        dict(lenient=True)),
        ("strict orphan-I",           dict(lenient=False)),

        ("lenient + bridge 2",        dict(lenient=True, bridge=2)),
        ("lenient + bridge 3",        dict(lenient=True, bridge=3)),
        ("lenient + bridge 4",        dict(lenient=True, bridge=4)),
        ("lenient + bridge 5",        dict(lenient=True, bridge=5)),

        ("strict + bridge 2",         dict(lenient=False, bridge=2)),
        ("strict + bridge 3",         dict(lenient=False, bridge=3)),
        ("strict + bridge 4",         dict(lenient=False, bridge=4)),
        ("strict + bridge 5",         dict(lenient=False, bridge=5)),

        ("strict + br3 + min3",       dict(lenient=False, bridge=3, min_chars=3)),
        ("strict + br4 + min3",       dict(lenient=False, bridge=4, min_chars=3)),
        ("lenient + br3 + min3",      dict(lenient=True, bridge=3, min_chars=3)),

        ("strict + br3 + min3 + c.3", dict(lenient=False, bridge=3,
                                           min_chars=3, min_conf=0.3)),
        ("strict + br3 + min3 + c.5", dict(lenient=False, bridge=3,
                                           min_chars=3, min_conf=0.5)),
        ("lenient + br2 + min3 + c.5", dict(lenient=True, bridge=2,
                                            min_chars=3, min_conf=0.5)),
    ]

    print(f"{'variant':<28} {'macroF1':>8} {'P':>7} {'R':>7} "
          f"{'microF1':>8} {'#pred':>7}")
    print("-" * 70)

    results = []
    for name, kw in variants:
        pred = {}
        for did in docs:
            if did not in saved:
                continue
            pred[did] = decode(saved[did]["tags"], offsets[did], docs[did],
                               conf=saved[did].get("conf"), **kw)
        mf, mp, mr, per, micro = strict_scores(pred, gold)
        n = sum(len(v) for v in pred.values())
        zeros = [l for l in per if per[l][2] == 0.0]
        print(f"{name:<28} {mf:>8.4f} {mp:>7.4f} {mr:>7.4f} "
              f"{micro[2]:>8.4f} {n:>7,}"
              + (f"   ZERO: {','.join(zeros)}" if zeros else ""))
        results.append((mf, name, pred, per))

    results.sort(reverse=True, key=lambda r: r[0])
    print(f"\n{'=' * 70}")
    print("per-label F1, top 3 variants")
    print(f"{'=' * 70}")
    labs = sorted(results[0][3], key=lambda l: results[0][3][l][2])
    print(f"  {'label':<12}" + "".join(f"{r[1][:16]:>18}" for r in results[:3]))
    for lab in labs:
        row = "".join(f"{r[3].get(lab, (0,0,0))[2]:>18.4f}" for r in results[:3])
        print(f"  {lab:<12}{row}")
    print(f"  {'MACRO':<12}" + "".join(f"{r[0]:>18.4f}" for r in results[:3]))


if __name__ == "__main__":
    main()