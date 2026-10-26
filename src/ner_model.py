"""
Token-classification head over a Phase 1 model.

The head itself is architecture-agnostic — a linear layer from d_model to 25
BIO tags. What differs per architecture is how bidirectional the encoder is,
and that difference is the central hypothesis of this phase:

  prefix_lm  - split_indices = T puts the WHOLE sequence in the prefix, so
               attention is fully bidirectional. The model saw splits of
               0.7-0.9 throughout pretraining, so this is a modest
               extrapolation rather than a new regime.
  encdec     - classify over the encoder output; the decoder is irrelevant here
  decoder    - causal: a token sees nothing to its right
  efficient  - causal AND windowed

PREDICTION (registered before any result): bidirectional architectures should
beat causal ones, and the gap should be largest for WIT and APP, which are
disambiguated by markers appearing AFTER the name ('(PW-6)', '(A-1)'). A
causal model cannot use them.
"""

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from crf import CRF

from bio_tagging import TAGS

N_TAGS = len(TAGS)          # 25
IGNORE = -100
N_POS_FEATS = 16

def position_features(doc_pos, device):
    """
    doc_pos: (B, T) float in [0, 1] — position within the DOCUMENT, not the
    window. Expanded into sinusoids at several scales so the head can use
    coarse location (header vs body) and finer gradations alike.

    A single scalar against 768 hidden dimensions is easy for a linear layer
    to ignore; a multi-scale encoding is not.
    """
    freqs = torch.tensor([2.0 ** i for i in range(N_POS_FEATS // 2)],
                         device=device)
    a = doc_pos.unsqueeze(-1) * freqs * math.pi      # (B, T, 8)
    return torch.cat([a.sin(), a.cos()], dim=-1)     # (B, T, 16)


class TokenClassifier(nn.Module):
    def __init__(self, base, arch, d_model, n_tags=N_TAGS, dropout=0.1,
                 use_position=False, use_crf = False):
        super().__init__()
        self.base = base
        self.arch = arch
        self.use_position = use_position
        self.dropout = nn.Dropout(dropout)
        self.use_crf = use_crf
        self.crf = CRF(n_tags) if use_crf else None

        in_dim = d_model + (N_POS_FEATS if use_position else 0)
        self.head = nn.Linear(in_dim, n_tags)

        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def hidden(self, input_ids, attn_mask=None):
        """(B, T, d_model) contextual representations."""
        B, T = input_ids.size()

        if self.arch == "prefix_lm":
            # whole sequence as prefix -> fully bidirectional
            splits = torch.full((B,), T, dtype=torch.long,
                                device=input_ids.device)
            return self.base(input_ids, attn_mask=attn_mask,
                             split_indices=splits, return_hidden=True)

        if self.arch == "encdec":
            return self.base.encode(input_ids, attn_mask)

        # decoder, efficient: causal
        return self.base(input_ids, attn_mask=attn_mask, return_hidden=True)

    def forward(self, input_ids, attn_mask=None, labels=None,
                class_weights=None, doc_pos = None):
        h = self.hidden(input_ids, attn_mask)
        if self.use_position:
            if doc_pos is None:
                raise ValueError("use_position=True requires doc_pos")
            h = torch.cat([h, position_features(doc_pos, h.device).to(h.dtype)], dim=-1)
        logits = self.head(self.dropout(h))

        loss = None
        if labels is not None:
            if self.use_crf:
                # CRF cannot ignore positions mid-sequence, so padded labels
                # are replaced by O and excluded via the mask instead
                mask = (labels != IGNORE)
                safe = labels.masked_fill(~mask, 0)
                loss = self.crf(logits.float(), safe, mask)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.reshape(-1),
                    weight=class_weights,
                    ignore_index=IGNORE,
                )
        return logits, loss

    def trainable_parameters(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        return t, a


def load_base(checkpoint, device):
    """Returns (model, arch, d_model) from a Phase 1 checkpoint."""
    from decoder_only_transformer import GPT, ModelConfig
    from prefix_gpt import PrefixGPT
    from prefix_lm import PrefixModelConfig
    from efficient_attn import EfficientGPT, EfficientModelConfig
    from encoder_decoder import EncoderDecoder, EncDecConfig

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "decoder")
    cfg = ck["config"]

    builder = {
        "decoder": (GPT, ModelConfig),
        "prefix_lm": (PrefixGPT, PrefixModelConfig),
        "efficient": (EfficientGPT, EfficientModelConfig),
        "encdec": (EncoderDecoder, EncDecConfig),
    }[arch]

    model = builder[0](builder[1](**cfg))
    model.load_state_dict(ck["model"])
    model.to(device)
    return model, arch, cfg["d_model"]