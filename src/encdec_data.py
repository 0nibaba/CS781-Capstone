"""
Batching for the encoder-decoder architecture.

Unlike the other three architectures, which return fixed (B, 1024) tensors,
span corruption produces VARIABLE-LENGTH sequences: encoder inputs around 880
tokens, decoder targets around 157 (and up to ~202 in long-terminal mode). So
batches must be padded, and the loss must ignore padded target positions.

DESIGN DECISIONS:

1. CORRUPTION HAPPENS ON THE FLY, per batch item. Pre-corrupting would mean
   every pass over the data sees the same corruption, which defeats the
   objective. The cost is ~6 spans x 32 items x 8 micro-batches of Python work
   per step; measured against a ~5 s step time this is negligible.

2. PAD TO THE BATCH MAXIMUM. Sorting by length to reduce padding waste would
   correlate batch composition with sequence length, which interacts badly
   with the source-mixture sampler. The waste is small because lengths are
   tightly clustered.

3. MATCHED ON TOKENS READ, NOT SUPERVISED POSITIONS. The encoder reads 1024
   tokens per sequence, the same as every other architecture, so the token
   budget is comparable. But only ~157 decoder positions carry loss, against
   1024 for the causal decoder — **roughly 6x less supervision per token
   processed**, worse than prefix-LM's 5x. Matching supervised positions
   instead would need ~6x the tokens, which the compute budget does not allow.
   The confound is real and is reported.
"""

import random
import numpy as np
import torch

from span_corruption import corrupt_sequence, EOS_ID

PAD_ID = EOS_ID          # <|endoftext|>; padded positions are masked out anyway


class EncDecBatcher:
    """Wraps a MemmapDataLoader and turns its raw windows into corrupted
    encoder/decoder batches."""

    def __init__(self, dataloader, sentinel_ids, p=0.15, mean_span=25,
                 p_terminal=0.25, p_long_terminal=0.15, seed=1899):
        self.dl = dataloader
        self.sentinels = sentinel_ids
        self.p = p
        self.mean_span = mean_span
        self.p_terminal = p_terminal
        self.p_long_terminal = p_long_terminal
        self.rng = random.Random(seed)
        self.eval_rng = random.Random(1337)      # fixed corruptions at eval

    def _corrupt_batch(self, windows, rng):
        enc, din, dtg = [], [], []
        for w in windows:
            e, i, t = corrupt_sequence(
                w, self.sentinels, p=self.p, mean_span=self.mean_span,
                p_terminal=self.p_terminal,
                p_long_terminal=self.p_long_terminal, rng=rng)
            enc.append(e); din.append(i); dtg.append(t)
        return enc, din, dtg

    @staticmethod
    def _pad(seqs, device):
        """-> (padded ids, mask) with mask 1 on real positions."""
        B = len(seqs)
        T = max(len(s) for s in seqs)
        ids = np.full((B, T), PAD_ID, dtype=np.int64)
        msk = np.zeros((B, T), dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            msk[i, :len(s)] = 1
        return (torch.from_numpy(ids).to(device, non_blocking=True),
                torch.from_numpy(msk).to(device, non_blocking=True))

    def get_batch(self, split, batch_size, device, source=None):
        """Returns enc_idx, enc_mask, dec_idx, dec_target, dec_mask."""
        # the underlying loader gives (x, y); only x is needed, as the window
        x, _ = self.dl.get_batch(split, batch_size, "cpu", source=source)
        windows = x.tolist()

        rng = self.eval_rng if split == "val" else self.rng

        enc, din, dtg = self._corrupt_batch(windows, rng)

        enc_idx, enc_mask = self._pad(enc, device)
        dec_idx, _ = self._pad(din, device)
        dec_tgt, dec_mask = self._pad(dtg, device)
        return enc_idx, enc_mask, dec_idx, dec_tgt, dec_mask


# ---------------- tests ----------------

if __name__ == "__main__":
    import os
    from tokenizers import Tokenizer
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import MemmapDataLoader

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok = Tokenizer.from_file(
        os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json"))
    sent = [tok.token_to_id(f"<|unused{i}|>") for i in range(10)]

    print("--- loading memmaps ---")
    dl = MemmapDataLoader(os.path.join(ROOT, "training_data/tokens"),
                          block_size=1024)
    b = EncDecBatcher(dl, sent)

    print("\n--- 1. shapes and padding ---")
    e, em, d, t, dm = b.get_batch("train", 8, "cpu")
    print(f"  enc_idx    {tuple(e.shape)}   real tokens "
          f"{em.sum().item()}/{em.numel()} ({100*em.float().mean():.1f}%)")
    print(f"  dec_idx    {tuple(d.shape)}")
    print(f"  dec_target {tuple(t.shape)}   supervised "
          f"{dm.sum().item()}/{dm.numel()} ({100*dm.float().mean():.1f}%)")
    assert d.shape == t.shape == dm.shape
    assert e.shape == em.shape

    sup = dm.sum().item()
    read = 8 * 1024
    print(f"\n  supervised positions {sup} for {read} tokens read "
          f"-> {read/sup:.1f}x less supervision than a causal decoder")

    print("\n--- 2. decoder input is the target shifted right ---")
    for i in range(8):
        n = int(dm[i].sum())
        assert d[i, 0] == EOS_ID, "decoder input must start with EOS"
        assert torch.equal(d[i, 1:n], t[i, :n-1]), f"shift wrong in row {i}"
    print("  all rows: dec_idx[1:n] == dec_target[:n-1], starting from EOS")

    print("\n--- 3. padded positions are marked, not silently valid ---")
    for i in range(8):
        n = int(em[i].sum())
        if n < e.shape[1]:
            assert (e[i, n:] == PAD_ID).all()
    print("  encoder padding uses PAD_ID beyond the mask")

    print("\n--- 4. eval is reproducible from a reset state ---")

    def fresh_val_batch():
        b.eval_rng.seed(1337)          # corruption RNG
        dl.eval_rng.manual_seed(1337)  # window-sampling RNG
        return b.get_batch("val", 4, "cpu", source="acts")

    a1, a2 = fresh_val_batch(), fresh_val_batch()
    same = all(torch.equal(x, y) for x, y in zip(a1, a2))
    print(f"  identical after resetting both RNGs: {same}")
    assert same, "eval is not reproducible; the val curve would carry noise"

    print("\n--- 5. readable sample ---")
    n = int(em[0].sum())
    enc_txt = tok.decode(e[0, :n].tolist(), skip_special_tokens=False)
    m = int(dm[0].sum())
    tgt_txt = tok.decode(t[0, :m].tolist(), skip_special_tokens=False)
    print(f"  encoder (first 400 chars):\n    {enc_txt[:400]}")
    print(f"\n  target (first 400 chars):\n    {tgt_txt[:400]}")