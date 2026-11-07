import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from crf import CRF

from bio_tagging import TAGS

N_TAGS = len(TAGS)
IGNORE = -100
N_POS_FEATS = 16

def position_features(doc_pos, device):
    freqs = torch.tensor([2.0 ** i for i in range(N_POS_FEATS // 2)],
                         device=device)
    a = doc_pos.unsqueeze(-1) * freqs * math.pi
    return torch.cat([a.sin(), a.cos()], dim=-1)


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
            splits = torch.full((B,), T, dtype=torch.long,
                                device=input_ids.device)
            return self.base(input_ids, attn_mask=attn_mask,
                             split_indices=splits, return_hidden=True)

        if self.arch == "encdec":
            return self.base.encode(input_ids, attn_mask)

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