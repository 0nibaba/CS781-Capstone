"""Phase 6 data: long documents -> long summaries, both over the 1024 ceiling.

Documents median 3,656 tokens (97% exceed 1024); summaries median 749. Neither
side fits, so the unit of training is a CHUNK PAIR: the document is split into
max_src-token chunks and the reference summary is split proportionally, giving
~5.4 pairs per document and ~38k pairs from 7,030 documents. At inference the
per-chunk summaries are concatenated in order.

This assumes summary content is ALIGNED with document order -- true for legal
judgments, where a summary follows facts -> arguments -> holding in the order
the judgment presents them. Stated because it is an assumption, not a fact, and
the chunk-alignment quality is what the hierarchical alternative would improve.

Empty summaries exist in train (min 0 tokens) and are dropped.
"""
import os, json, random
import torch
from summ_model import BOS_ID, EOS_ID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training_data/raw/Phase_6_Summ/Train")
TEST_DIR = os.path.join(ROOT, "test_data/Phase_6_Summ/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

MAX_SRC = 1024
MAX_TGT = 320
PAD_ID = 0
EOS_ID = 0
DEV_N = 500
SPLIT_SEED = 1899
CAUSAL_SRC = 480
CAUSAL_TGT = 512


def collate(chunk, device, arch, max_src=MAX_SRC, max_tgt=512):
    if arch == "encdec":
        ns = max(len(it["src"]) for it in chunk)
        nt = max(len(it["tgt"]) for it in chunk) + 1
        enc = torch.full((len(chunk), ns), PAD_ID, dtype=torch.long)
        em = torch.zeros(len(chunk), ns, dtype=torch.long)
        din = torch.full((len(chunk), nt), PAD_ID, dtype=torch.long)
        dout = torch.full((len(chunk), nt), PAD_ID, dtype=torch.long)
        dm = torch.zeros(len(chunk), nt, dtype=torch.long)
        for i, it in enumerate(chunk):
            s, t = it["src"], it["tgt"][:max_tgt]
            enc[i, :len(s)] = torch.tensor(s); em[i, :len(s)] = 1
            seq = [BOS_ID] + t + [EOS_ID]
            din[i, :len(seq) - 1] = torch.tensor(seq[:-1])
            dout[i, :len(seq) - 1] = torch.tensor(seq[1:])
            dm[i, :len(seq) - 1] = 1
        return {"enc_idx": enc.to(device), "enc_mask": em.to(device),
                "dec_in": din.to(device), "dec_out": dout.to(device),
                "dec_mask": dm.to(device)}

    rows, splits = [], []
    for it in chunk:
        s = it["src"][:CAUSAL_SRC]
        t = it["tgt"][:CAUSAL_TGT]
        rows.append(s + [EOS_ID] + t + [EOS_ID])
        splits.append(len(s) + 1)
    n = max(len(r) for r in rows)
    ids = torch.full((len(rows), n), PAD_ID, dtype=torch.long)
    am = torch.zeros(len(rows), n, dtype=torch.long)
    tg = torch.full((len(rows), n), -100, dtype=torch.long)
    for i, (r, sp) in enumerate(zip(rows, splits)):
        ids[i, :len(r)] = torch.tensor(r); am[i, :len(r)] = 1
        tg[i, sp:len(r)] = torch.tensor(r[sp:])     # loss on summary only
    return {"ids": ids.to(device), "attn_mask": am.to(device),
            "targets": tg.to(device),
            "split": torch.tensor(splits, device=device)}


def batches(ds, batch_size, device, arch, shuffle=True, seed=0,
            chunk0_only=False):
    items = [i for i, it in enumerate(ds.items)
             if (not chunk0_only or it["chunk"] == 0)]
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        items = [items[i] for i in torch.randperm(len(items),
                                                  generator=g).tolist()]
    for i in range(0, len(items), batch_size):
        c = [ds.items[j] for j in items[i:i + batch_size]]
        yield c, collate(c, device, arch)


def load_raw(split):
    path = (os.path.join(TEST_DIR, "test.json") if split == "test"
            else os.path.join(TRAIN_DIR, "train.json"))
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "document": v}
             for k, v in d.items()]
    return d


def load_split(split):
    """split in {train, dev, test}. train/dev are a deterministic partition."""
    if split == "test":
        return load_raw("test")
    docs = [d for d in load_raw("train")
            if d.get("summary") and d["summary"].strip()
            and d.get("document") and d["document"].strip()]
    docs.sort(key=lambda d: str(d["id"]))          # stable before shuffling
    rng = random.Random(SPLIT_SEED)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    dev_idx = set(idx[:DEV_N])
    keep = [docs[i] for i in range(len(docs))
            if (i in dev_idx) == (split == "dev")]
    return keep


def chunk_pairs(doc_ids, sum_ids, max_src=MAX_SRC, max_tgt=MAX_TGT):
    """Split a document into chunks and its summary proportionally.

    The summary is divided in the same ratio as the document, so chunk i gets
    the slice of the summary that corresponds to its share of the source.
    """
    n_chunks = max(1, -(-len(doc_ids) // max_src))
    per_tgt = max(1, -(-len(sum_ids) // n_chunks)) if sum_ids else 0
    out = []
    for c in range(n_chunks):
        src = doc_ids[c * max_src:(c + 1) * max_src]
        if len(src) < 32 and out:
            break
        tgt = sum_ids[c * per_tgt:(c + 1) * per_tgt][:max_tgt] if sum_ids else []
        out.append((src, tgt, c, n_chunks))
    return out


class SummPairs:
    """Chunk pairs for training; one entry per (document chunk, summary slice)."""

    def __init__(self, docs, tok, max_src=MAX_SRC, max_tgt=MAX_TGT,
                 batch=64, with_target=True):
        self.items = []
        for i in range(0, len(docs), batch):
            part = docs[i:i + batch]
            denc = tok.encode_batch([d["document"] for d in part])
            senc = (tok.encode_batch([d["summary"] for d in part])
                    if with_target else [None] * len(part))
            for d, de, se in zip(part, denc, senc):
                sid = se.ids if se is not None else []
                for src, tgt, c, n in chunk_pairs(de.ids, sid,
                                                  max_src, max_tgt):
                    self.items.append({
                        "id": d["id"], "src": src, "tgt": tgt,
                        "chunk": c, "n_chunks": n,
                        "n_sum_tokens": len(sid),
                    })

    def __len__(self):
        return len(self.items)