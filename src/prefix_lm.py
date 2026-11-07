import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

SPLIT_LO = 0.7
SPLIT_HI = 0.9
SPLIT_EVAL = 0.8 


@dataclass
class PrefixModelConfig:
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = False
    use_internal_causal_mask: bool = False
    doc_mask: bool = False
    eos_id: int = 0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(ms + self.eps).type_as(x))


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_head
        self.dropout = config.dropout
        self.use_internal_causal_mask = config.use_internal_causal_mask

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_internal_causal_mask:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size,
                                      dtype=torch.bool))
                     .view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )

        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, attn_mask=None, force_manual=False):
        B, T, C = x.size()

        if not self.use_internal_causal_mask and attn_mask is None:
            raise ValueError(
                "use_internal_causal_mask=False requires an explicit attn_mask; "
                "without one the model would attend to future positions.")

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if self.use_internal_causal_mask:
            keep = self.causal_mask[:, :, :T, :T]
            if attn_mask is not None:
                keep = keep & attn_mask
        else:
            keep = attn_mask

        if self.use_flash and not force_manual:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            att = att.masked_fill(~keep, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


def sample_split_indices(B, T, device, generator=None):
    """Uniform over [0.7T, 0.9T]; see design note 1."""
    lo, hi = int(SPLIT_LO * T), int(SPLIT_HI * T)
    return torch.randint(lo, hi + 1, (B,), device=device, generator=generator)


def build_prefix_lm_mask(B, T, split_indices, device, pad_mask=None):
    s = split_indices.view(B, 1, 1, 1)
    rows = torch.arange(T, device=device).view(1, 1, T, 1)
    cols = torch.arange(T, device=device).view(1, 1, 1, T)

    key_is_prefix = cols < s
    both_in_suffix = (rows >= s) & (cols >= s)
    causal = rows >= cols

    keep = key_is_prefix | (both_in_suffix & causal)

    if pad_mask is not None:
        keep = keep & pad_mask.bool().view(B, 1, 1, T)
    return keep


def suffix_loss_mask(B, T, split_indices, device):
    rows = torch.arange(T, device=device).view(1, T)
    s = split_indices.view(B, 1)
    return (rows >= (s - 1)) & (rows <= T - 2)


def compute_suffix_loss(logits, targets, split_indices):
    B, T, V = logits.shape
    per_tok = F.cross_entropy(
        logits.view(-1, V).float(), targets.reshape(-1), reduction="none"
    ).view(B, T)

    m = suffix_loss_mask(B, T, split_indices, logits.device).float()
    return (per_tok * m).sum() / m.sum().clamp(min=1.0)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x, attn_mask=None, force_manual_attn=False):
        x = x + self.attn(self.norm1(x), attn_mask, force_manual_attn)
        x = x + self.ffn(self.norm2(x))
        return x