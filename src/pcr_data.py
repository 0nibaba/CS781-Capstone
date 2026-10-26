"""PCR data: cases as sequences of 1024-token chunks over the first 4096.

BM25 peaks at a 4,096-token budget (0.2991) and decays monotonically beyond it
(full document 0.0734), so the first ~4 windows hold the discriminative content
and the rest is boilerplate noise. The dense model is capped at 1024 tokens by
Phase 1's learned positions, so a case is represented as up to 4 chunks and a
query-candidate pair is scored by MAX similarity over chunk pairs.

Max rather than mean, for the same reason max-pooling won in Phase 5: relevance
between two judgments often turns on one matching passage, and averaging
dilutes it with chunks that correctly report no match.

Training samples ONE chunk from the query and one from a relevant candidate per
example, so the model is trained on exactly the chunk-level matching that
max-sim inference performs.
"""
import os, json, random
import torch

from pcr_baseline import load_split, TOKENIZER   # shared loader

MAX_DOC = 4096
CHUNK = 1024
PAD_ID = 0


def chunk_ids(ids, chunk=CHUNK, max_doc=MAX_DOC):
    ids = ids[:max_doc]
    out = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    out = [c for c in out if len(c) >= 32] or [ids[:chunk] or [PAD_ID]]
    return out


class Corpus:
    """Pre-tokenised chunks for every document in a split."""

    def __init__(self, texts, tok, batch=64):
        self.ids = list(texts)
        self.chunks = {}
        keys = self.ids
        for i in range(0, len(keys), batch):
            part = keys[i:i + batch]
            encs = tok.encode_batch([texts[k] for k in part])
            for k, e in zip(part, encs):
                self.chunks[k] = chunk_ids(e.ids)

    def flat(self):
        """[(doc_id, chunk_ids), ...] for embedding the whole pool."""
        return [(k, c) for k in self.ids for c in self.chunks[k]]


def pack(chunks, device):
    n = max(len(c) for c in chunks)
    I = torch.full((len(chunks), n), PAD_ID, dtype=torch.long)
    M = torch.zeros(len(chunks), n, dtype=torch.long)
    for r, c in enumerate(chunks):
        I[r, :len(c)] = torch.tensor(c, dtype=torch.long)
        M[r, :len(c)] = 1
    return I.to(device), M.to(device)


class PairSampler:
    """(query, positive-candidate) pairs; one chunk sampled from each."""

    def __init__(self, queries, qcorpus, ccorpus, seed=1899):
        self.pairs = [(q, c) for q, v in queries.items()
                      for c in v["gold"] if c in ccorpus.chunks]
        self.q, self.c = qcorpus, ccorpus
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def batches(self, batch_size, device, shuffle=True, epoch=0,
                hard=None):
        idx = list(range(len(self.pairs)))
        if shuffle:
            random.Random(1899 + epoch).shuffle(idx)
        for i in range(0, len(idx), batch_size):
            sel = [self.pairs[j] for j in idx[i:i + batch_size]]
            # one query per batch slot; duplicates would make in-batch
            # negatives false, so drop repeats within a batch
            seen, keep = set(), []
            for q, c in sel:
                if q in seen:
                    continue
                seen.add(q); keep.append((q, c))
            if len(keep) < 2:
                continue
            qc = [self.rng.choice(self.q.chunks[q]) for q, _ in keep]
            cc = [self.rng.choice(self.c.chunks[c]) for _, c in keep]
            neg = None
            if hard is not None:
                negs = []
                for q, _ in keep:
                    pool = hard.get(q, [])
                    negs.append(self.rng.choice(self.c.chunks[
                        self.rng.choice(pool)]) if pool else None)
                if all(n is not None for n in negs):
                    neg = pack(negs, device)
            yield keep, pack(qc, device), pack(cc, device), neg