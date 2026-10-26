"""Multi-label statute classifier: 100 independent sigmoids (binary relevance).

Reuses DocClassifier's architecture dispatch and pooling; only the head width
and the loss change. BCE with an optional per-label pos_weight, which is the
standard handling for positive/negative imbalance -- each statute is positive in
0.3%-27% of documents, so the negatives overwhelm the positives for most labels.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from cjpe_model import DocClassifier, POOLINGS
from lsi_data import N_LABELS


class StatuteClassifier(DocClassifier):
    def __init__(self, base, arch, d_model, n_labels=N_LABELS, dropout=0.1,
                 pooling="mean"):
        super().__init__(base, arch, d_model, n_labels=n_labels,
                         dropout=dropout, pooling=pooling)

    def forward(self, input_ids, attn_mask, labels=None, pos_weight=None):
        h = self.hidden(input_ids, attn_mask)
        v = self.pool(h, attn_mask)
        logits = self.head(self.dropout(v))          # (B, N_LABELS)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), labels.float(), pos_weight=pos_weight)
        return logits, loss