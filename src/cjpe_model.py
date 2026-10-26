"""Document classifier over a Phase 1 model, for binary judgment prediction.

A judgment is 2.7k-3.9k tokens against a 1024-token ceiling, so the model sees
a WINDOW of the case (cjpe_data.select picks which) and must reduce that window
to one vector. Three pooling strategies, because the right one depends on the
architecture:

  last  -- the final non-pad position. Under causal attention this is the only
           position that has read the entire window. Near-useless for a
           bidirectional encoder, where every position already sees everything.
  mean  -- masked average over the window. The natural default for prefix_lm
           and encdec; discards position entirely.
  attn  -- a learned query scores every position and pools by that. Lets the
           model decide which parts of a judgment carry the outcome, which is
           the question the brief asks directly.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from ner_model import load_base          # arch dispatch, reused since Phase 2

POOLINGS = ("last", "mean", "attn")


class DocClassifier(nn.Module):
    def __init__(self, base, arch, d_model, n_labels=2, dropout=0.1,
                 pooling="last"):
        super().__init__()
        assert pooling in POOLINGS, pooling
        self.base = base
        self.arch = arch
        self.pooling = pooling

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_labels)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

        if pooling == "attn":
            self.score = nn.Linear(d_model, 1)
            nn.init.normal_(self.score.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.score.bias)

    def hidden(self, input_ids, attn_mask):
        B, T = input_ids.size()
        if self.arch == "prefix_lm":
            splits = torch.full((B,), T, dtype=torch.long,
                                device=input_ids.device)
            return self.base(input_ids, attn_mask=attn_mask,
                             split_indices=splits, return_hidden=True)
        if self.arch == "encdec":
            return self.base.encode(input_ids, attn_mask)
        return self.base(input_ids, attn_mask=attn_mask, return_hidden=True)

    def pool(self, h, attn_mask):
        """(B,T,d) -> (B,d)"""
        m = attn_mask.bool()
        if self.pooling == "last":
            idx = attn_mask.sum(1) - 1                      # (B,)
            idx = idx.clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
            return h.gather(1, idx).squeeze(1)
        if self.pooling == "mean":
            w = m.unsqueeze(-1).to(h.dtype)
            return (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        s = self.score(h).squeeze(-1)                       # (B,T)
        s = s.masked_fill(~m, float("-inf"))
        a = s.softmax(-1).unsqueeze(-1)
        return (h * a).sum(1)

    def forward(self, input_ids, attn_mask, labels=None, class_weights=None):
        h = self.hidden(input_ids, attn_mask)
        v = self.pool(h, attn_mask)
        logits = self.head(self.dropout(v))                 # (B, n_labels)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels,
                                   weight=class_weights)
        return logits, loss

    def trainable_parameters(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        return t, a


if __name__ == "__main__":
    import os, math
    from tokenizers import Tokenizer
    from cjpe_data import CJPEDocs, load_split, batches, TOKENIZER

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT = os.path.join(ROOT, "checkpoints/arm_b_docmask/ckpt_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.from_file(TOKENIZER)
    docs = load_split("dev")[:32]
    base, arch, d_model = load_base(CKPT, device)

    for mode in ("head", "tail", "head_tail"):
        ds = CJPEDocs(docs, tok, mode=mode)
        for pooling in POOLINGS:
            m = DocClassifier(base, arch, d_model, pooling=pooling).to(device)
            chunk, (ids, msk, lab) = next(iter(batches(ds, 8, device, seed=0)))
            logits, loss = m(ids, msk, labels=lab)
            print(f"  {mode:10s} {pooling:5s} logits {tuple(logits.shape)} "
                  f"loss {loss.item():.4f} (ln2 = {math.log(2):.4f})")
            assert logits.shape == (8, 2)

    print("\n--- last-token pooling reads the final real token, not padding ---")
    ds = CJPEDocs(docs, tok, mode="tail")
    m = DocClassifier(base, arch, d_model, pooling="last").to(device).eval()
    chunk, (ids, msk, lab) = next(iter(batches(ds, 8, device, seed=0)))
    with torch.no_grad():
        h = m.hidden(ids, msk)
        v = m.pool(h, msk)
    b = 0
    last = int(msk[b].sum()) - 1
    assert torch.allclose(v[b], h[b, last])
    print(f"  ok: pooled vector == h[{last}] for a window of "
          f"{int(msk[b].sum())} tokens")