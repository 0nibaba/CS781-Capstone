from collections import defaultdict

from bio_tagging import LABELS


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def strict_scores(pred, gold, labels=LABELS):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for did in set(pred) | set(gold):
        p = {(e[0], e[1], e[2]) for e in pred.get(did, [])}
        g = {(e[0], e[1], e[2]) for e in gold.get(did, [])}
        for e in p & g:
            tp[e[2]] += 1
        for e in p - g:
            fp[e[2]] += 1
        for e in g - p:
            fn[e[2]] += 1

    per = {}
    for lab in labels:
        if tp[lab] + fp[lab] + fn[lab] == 0:
            continue
        per[lab] = prf(tp[lab], fp[lab], fn[lab]) + (
            tp[lab], fp[lab], fn[lab])

    if not per:
        return 0.0, 0.0, 0.0, {}, (0.0, 0.0, 0.0)

    macro_p = sum(v[0] for v in per.values()) / len(per)
    macro_r = sum(v[1] for v in per.values()) / len(per)
    macro_f = sum(v[2] for v in per.values()) / len(per)

    micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return macro_f, macro_p, macro_r, per, micro


def report(pred, gold, title=""):
    mf, mp, mr, per, micro = strict_scores(pred, gold)
    if title:
        print(f"\n{title}")
    print(f"  strict macro-F1 {mf:.4f}  (P {mp:.4f} / R {mr:.4f})")
    print(f"  strict micro-F1 {micro[2]:.4f}  (P {micro[0]:.4f} / "
          f"R {micro[1]:.4f})")
    print(f"\n  {'label':<12} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'tp':>6} {'fp':>6} {'fn':>6}")
    print("  " + "-" * 56)
    for lab in sorted(per, key=lambda l: per[l][2]):
        p, r, f, t, fpv, fnv = per[lab]
        print(f"  {lab:<12} {p:>7.4f} {r:>7.4f} {f:>7.4f} "
              f"{t:>6} {fpv:>6} {fnv:>6}")
    return mf


if __name__ == "__main__":
    import os, json
    from tokenizers import Tokenizer
    from bio_tagging import spans_to_bio, bio_to_spans

    tok = Tokenizer.from_file(TOKENIZER if False else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints/tokenizers/legal_32000.json"))

    base = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "training_data/raw/Phase_2_LNER/Train")
    docs = json.load(open(os.path.join(base, "fold1-text.json")))
    gold = json.load(open(os.path.join(base, "fold1-gold.json")))
    gold = {k: [[e[0], e[1], e[2]] for e in v] for k, v in gold.items()}

    print("--- 1. gold against itself ---")
    mf = report(gold, gold)
    assert abs(mf - 1.0) < 1e-9, "metric does not return 1.0 on exact match"

    print("\n--- 2. gold round-tripped through BIO ---")
    rt = {}
    for did, text in docs.items():
        enc = tok.encode(text)
        tags, _ = spans_to_bio(gold[did], enc.offsets, len(enc.ids))
        rt[did] = bio_to_spans(tags, enc.offsets, text)
    mf = report(rt, gold)
    print(f"\n  expect ~0.98-0.99: this is the alignment ceiling, not a model")