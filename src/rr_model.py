"""
Sentence classifier over a Phase 1 model, with an optional CRF over the
document's label sequence.

Sentence representations are taken from the separator token appended after
each sentence. For a causal model that token has read the whole sentence plus
everything before it; for a bidirectional one it sees the document both ways.
Mean-pooling over each sentence's own tokens is available as an alternative,
since the brief asks about sentence representations.

The CRF is unconstrained: unlike BIO in Phase 2, every rhetorical-role
transition is structurally possible. Its value here comes from the 94% / 75%
self-transition rate, which independent per-sentence classification cannot
exploit at all.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from crf import CRF
from ner_model import load_base          # arch dispatch, reused from Phase 2
from rr_data import N_LABELS, IGNORE


class SentenceClassifier(nn.Module):
    def __init__(self, base, arch, d_model, n_labels=N_LABELS, dropout=0.1,
                 use_crf=False, pooling="marker"):
        super().__init__()
        self.base = base
        self.arch = arch
        self.pooling = pooling
        self.use_crf = use_crf

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_labels)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.crf = CRF(n_labels, constrain=False) if use_crf else None

    def hidden(self, input_ids, attn_mask):
        B, T = input_ids.size()

        if self.arch == "prefix_lm":
            # whole block as prefix -> fully bidirectional
            splits = torch.full((B,), T, dtype=torch.long,
                                device=input_ids.device)
            return self.base(input_ids, attn_mask=attn_mask,
                             split_indices=splits, return_hidden=True)
        if self.arch == "encdec":
            return self.base.encode(input_ids, attn_mask)
        return self.base(input_ids, attn_mask=attn_mask, return_hidden=True)

    def sentence_vectors(self, h, marker_pos):
        """(B,T,d) -> (B,S,d), one vector per predicted sentence."""
        idx = marker_pos.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return h.gather(1, idx)

    def forward(self, input_ids, attn_mask, marker_pos, marker_mask,
                labels=None, class_weights=None):
        h = self.hidden(input_ids, attn_mask)
        s = self.sentence_vectors(h, marker_pos)        # (B, S, d)
        logits = self.head(self.dropout(s))             # (B, S, n_labels)

        loss = None
        if labels is not None:
            if self.use_crf:
                # the CRF cannot skip positions mid-sequence, so padded slots
                # are given label 0 and excluded through the mask instead
                m = marker_mask.bool() & (labels != IGNORE)
                safe = labels.masked_fill(~m, 0)
                loss = self.crf(logits.float(), safe, m)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.reshape(-1),
                    weight=class_weights,
                    ignore_index=IGNORE,
                )
        return logits, loss

    @torch.no_grad()
    def decode(self, logits, marker_mask):
        """Per-block label sequences, as a list of lists."""
        if self.use_crf:
            return self.crf.viterbi(logits.float(), marker_mask)
        pred = logits.argmax(-1).cpu()
        return [pred[b, :int(marker_mask[b].sum())].tolist()
                for b in range(pred.size(0))]

    def trainable_parameters(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        return t, a


# ---------------- tests ----------------

if __name__ == "__main__":
    import os
    from tokenizers import Tokenizer
    from rr_data import (RRBlocks, load_split, batches, TOKENIZER, LABELS)

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT = os.path.join(ROOT, "checkpoints/arm_b_docmask/ckpt_best.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(TOKENIZER)
    ds = RRBlocks(load_split("dev"), tok)

    base, arch, d_model = load_base(CKPT, device)

    for use_crf in (False, True):
        m = SentenceClassifier(base, arch, d_model, use_crf=use_crf).to(device)
        t, a = m.trainable_parameters()
        print(f"\n--- crf={use_crf} | arch {arch} | trainable {t:,} ---")

        chunk, (ids, msk, pos, pmask, lab) = next(
            iter(batches(ds, 4, device, seed=0)))
        logits, loss = m(ids, msk, pos, pmask, labels=lab)
        print(f"  logits {tuple(logits.shape)}  loss {loss.item():.4f}")
        if not use_crf:
            import math
            print(f"  (cross-entropy at init should be near "
                  f"ln({N_LABELS}) = {math.log(N_LABELS):.4f})")

        paths = m.decode(logits, pmask)
        lens_ok = all(len(p) == int(pmask[b].sum())
                      for b, p in enumerate(paths))
        print(f"  decoded {len(paths)} blocks, lengths match the mask: {lens_ok}")
        assert lens_ok
        assert all(0 <= x < N_LABELS for p in paths for x in p)

    print("\n--- sentence vectors come from distinct positions ---")
    m = SentenceClassifier(base, arch, d_model).to(device).eval()
    with torch.no_grad():
        h = m.hidden(ids, msk)
        s = m.sentence_vectors(h, pos)
    # vector j must equal the hidden state at marker position j
    b, j = 0, 3
    same = torch.allclose(s[b, j], h[b, pos[b, j]])
    print(f"  gathered vector matches h at its marker position: {same}")
    assert same
    d = (s[0, 0] - s[0, 1]).abs().max().item()
    print(f"  consecutive sentence vectors differ by {d:.3f} (should be > 0)")