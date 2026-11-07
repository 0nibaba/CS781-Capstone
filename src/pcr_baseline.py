import os, json, math, argparse
from collections import Counter

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {
    "train": (os.path.join(ROOT, "training_data/raw/Phase_7_PCR/Train"),
              "train.json", "train_candidates.json"),
    "dev": (os.path.join(ROOT, "training_data/raw/Phase_7_PCR/Dev"),
            "dev.json", "dev_candidates.json"),
    "test": (os.path.join(ROOT, "test_data/Phase_7_PCR/Test"),
             "test.json", "test_candidates.json"),
}
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")


def as_text(v):
    if isinstance(v, dict):
        v = v.get("text", v.get("document", ""))
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def load_split(split):
    d, qf, cf = DIRS[split]
    with open(os.path.join(d, qf)) as f:
        Q = json.load(f)
    with open(os.path.join(d, cf)) as f:
        C = json.load(f)
    queries = {k: {"text": as_text(v),
                   "gold": list(v.get("relevant_candidates", []))
                   if isinstance(v, dict) else []}
               for k, v in Q.items()}
    cands = {k: as_text(v) for k, v in C.items()}
    return queries, cands


def micro_f1_at_k(ranked, queries, k=7):
    tp = sum(len(set(ranked[q][:k]) & set(v["gold"]))
             for q, v in queries.items())
    n_pred = k * len(queries)
    n_gold = sum(len(v["gold"]) for v in queries.values())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def ceiling(queries, k=7):
    tp = sum(min(len(v["gold"]), k) for v in queries.values())
    p = tp / (k * len(queries))
    r = tp / sum(len(v["gold"]) for v in queries.values())
    return 2 * p * r / (p + r), p, r


def bm25_rank(queries, cands, tok, k1=1.2, b=0.75, max_tokens=0):
    def toks(t):
        ids = tok.encode(t).ids
        return ids[:max_tokens] if max_tokens else ids

    cids = list(cands)
    docs = [Counter(toks(cands[c])) for c in cids]
    dl = np.array([sum(d.values()) for d in docs], dtype=np.float64)
    avgdl = dl.mean()
    df = Counter()
    for d in docs:
        df.update(d.keys())
    N = len(docs)
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    inv = {}
    for i, d in enumerate(docs):
        for t, f in d.items():
            inv.setdefault(t, []).append((i, f))

    out = {}
    for q, v in queries.items():
        qt = Counter(toks(v["text"]))
        s = np.zeros(N)
        for t, qf in qt.items():
            if t not in inv:
                continue
            w = idf[t]
            for i, f in inv[t]:
                s[i] += w * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * dl[i] / avgdl))
        order = np.argsort(-s)
        out[q] = [cids[i] for i in order]
    return out


@torch.no_grad()
def dense_rank(queries, cands, tok, ckpt, device, mode="head", max_len=1024,
               batch=8):
    from cjpe_model import DocClassifier
    from ner_model import load_base
    base, arch, d_model = load_base(ckpt, device)
    m = DocClassifier(base, arch, d_model, pooling="mean").to(device).eval()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def embed(texts):
        E = []
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            ids_l = []
            for t in part:
                ids = tok.encode(t).ids
                if len(ids) > max_len:
                    ids = (ids[:max_len] if mode == "head"
                           else ids[-max_len:] if mode == "tail"
                           else ids[:max_len // 2] + ids[-(max_len // 2):])
                ids_l.append(ids)
            n = max(len(x) for x in ids_l)
            I = torch.zeros(len(ids_l), n, dtype=torch.long)
            M = torch.zeros(len(ids_l), n, dtype=torch.long)
            for r, x in enumerate(ids_l):
                I[r, :len(x)] = torch.tensor(x); M[r, :len(x)] = 1
            with ctx:
                h = m.hidden(I.to(device), M.to(device))
                v = m.pool(h, M.to(device))
            E.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu())
        return torch.cat(E)

    cids = list(cands)
    CE = embed([cands[c] for c in cids])
    qids = list(queries)
    QE = embed([queries[q]["text"] for q in qids])
    S = QE @ CE.T
    return {q: [cids[i] for i in torch.argsort(-S[r]).tolist()]
            for r, q in enumerate(qids)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--bm25_tokens", type=int, nargs="+", default=[0, 1024])
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    queries, cands = load_split(args.split)
    print(f"{args.split}: {len(queries)} queries, {len(cands)} candidates")
    c = ceiling(queries)
    print(f"perfect-ranking ceiling: F1 {c[0]:.4f} (P {c[1]:.4f} R {c[2]:.4f})")
    print(f"\n  {'method':<28} {'F1@7':>8} {'P@7':>8} {'R@7':>8}")

    for mt in args.bm25_tokens:
        r = bm25_rank(queries, cands, tok, max_tokens=mt)
        f, pr, rc = micro_f1_at_k(r, queries)
        tag = "full doc" if mt == 0 else f"first {mt}"
        print(f"  {'BM25 ' + tag:<28} {f:>8.4f} {pr:>8.4f} {rc:>8.4f}")

    if args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for mode in ("head", "tail", "head_tail"):
            r = dense_rank(queries, cands, tok, args.checkpoint, device, mode)
            f, pr, rc = micro_f1_at_k(r, queries)
            print(f"  {'zero-shot dense ' + mode:<28} {f:>8.4f} "
                  f"{pr:>8.4f} {rc:>8.4f}")


if __name__ == "__main__":
    main()