"""Bi-encoder over a Phase 1 model: a chunk -> one L2-normalised vector."""
import torch
import torch.nn as nn
from torch.nn import functional as F

from cjpe_model import DocClassifier


class CaseEncoder(DocClassifier):
    def __init__(self, base, arch, d_model, pooling="mean", dropout=0.0,
                 proj_dim=0):
        super().__init__(base, arch, d_model, n_labels=2, dropout=dropout,
                         pooling=pooling)
        del self.head                       # no classifier here
        self.proj = (nn.Linear(d_model, proj_dim, bias=False)
                     if proj_dim else None)

    def embed(self, ids, mask):
        h = self.hidden(ids, mask)
        v = self.pool(h, mask)
        if self.proj is not None:
            v = self.proj(v)
        return F.normalize(v.float(), dim=-1)

    def info_nce(self, qv, cv, temp=0.05, neg=None):
        """In-batch negatives; optional one hard negative per query."""
        logits = qv @ cv.T / temp                       # (B, B)
        if neg is not None:
            extra = (qv * neg).sum(-1, keepdim=True) / temp
            logits = torch.cat([logits, extra], 1)      # (B, B+1)
        target = torch.arange(qv.size(0), device=qv.device)
        return F.cross_entropy(logits, target)