"""Lead-k extractive baseline, and the local BERTScore harness.

Reference summaries share 86.1% of their token types with the source document,
so on BERTScore -- which compares contextual embeddings, not exact strings --
simply returning the opening of the judgment is a serious competitor. This
establishes the floor. Any trained model that does not clear it is not earning
its keep, and reporting that honestly is the point.

Length is matched to the reference distribution (median 749 tokens), because
BERTScore F1 punishes summaries that are much shorter than the reference
through recall.
"""
import os, json, argparse
import numpy as np
from tokenizers import Tokenizer
from bert_score import BERTScorer

from summ_data import load_split, TOKENIZER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lead_k(text, tok, k_tokens):
    ids = tok.encode(text).ids[:k_tokens]
    return tok.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--bert_model", default="")
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--k", type=int, nargs="+", default=[256, 512, 750, 1000])
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    docs = load_split(args.split)[:args.limit]
    refs = [d["summary"] for d in docs]
    print(f"{args.split}: {len(docs)} documents")
    rl = np.array([len(tok.encode(r).ids) for r in refs])
    print(f"reference length: median {np.median(rl):.0f} tokens")

    kw = {"lang": "en", "rescale_with_baseline": False}
    if args.bert_model:
        kw = {"model_type": args.bert_model, "num_layers": 17,
              "rescale_with_baseline": False}
    scorer = BERTScorer(**kw)

    print(f"\n  {'baseline':<16} {'len':>6} {'BERTScore F1':>14}")
    for k in args.k:
        cands = [lead_k(d["document"], tok, k) for d in docs]
        P, R, F = scorer.score(cands, refs)
        ln = np.mean([len(tok.encode(c).ids) for c in cands])
        print(f"  lead-{k:<11} {ln:>6.0f} {F.mean().item():>14.4f}")

    # the other trivial baseline: the whole document
    cands = [d["document"] for d in docs]
    P, R, F = scorer.score(cands, refs)
    print(f"  {'full document':<16} {'-':>6} {F.mean().item():>14.4f}")


if __name__ == "__main__":
    main()