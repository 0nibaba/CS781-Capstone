"""
Architecture 3: local/global interleaved decoder.

Most layers attend within a causal window of W tokens; a few layers attend
over the full sequence so information still propagates globally.

DESIGN NOTE — this is NOT expected to be faster, and measuring that is the
point. Implementing a window as a boolean mask passed to SDPA still computes
the full (T,T) score matrix and then discards most of it: zero FLOPs saved.
Worse, an explicit attn_mask disables SDPA's FlashAttention kernel, which was
measured at a ~21% end-to-end cost for any per-sequence 4D mask. So the
masked implementation should be SLOWER than full causal attention despite
attending to less.

Real savings need block-sparse kernels (FlexAttention under torch.compile) or
chunked computation. Not attempted here; reported as untested future work.
The three attention_mode values exist so that claim can be measured rather
than asserted.

EFFECTIVE RECEPTIVE FIELD: a window of W=256 does not mean a token only ever
sees 256 positions. Each stacked windowed layer extends the field by W, so
after k consecutive windowed layers a token effectively sees ~k*W positions.
With global layers at (3, 7, 11) the longest windowed run is 3 layers, giving
an effective field of ~768 tokens before a global layer mixes everything.
"Local" is considerably less local than the window size suggests.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field

from decoder_only_transformer import GPT, ModelConfig, RMSNorm, FeedForward


@dataclass
class EfficientModelConfig(ModelConfig):
    window_size: int = 256          # radius: a token sees itself + W previous
    global_layers: tuple = (3, 7, 11)
    attention_mode: str = "masked"  # "masked" | "global_baseline"


def build_window_mask(T, W, device):
    """(1, 1, T, T) causal band: keep j where 0 <= i - j <= W."""
    rows = torch.arange(T, device=device).view(T, 1)
    cols = torch.arange(T, device=device).view(1, T)
    keep = (rows >= cols) & ((rows - cols) <= W)
    return keep.view(1, 1, T, T)


class WindowedSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_head
        self.dropout = config.dropout

        self.is_global = layer_idx in tuple(config.global_layers)
        self.window_size = config.window_size
        self.mode = config.attention_mode

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # one band mask per layer, cached; sliced to T at use
        if not self.is_global and self.mode == "masked":
            self.register_buffer(
                "window_mask",
                build_window_mask(config.block_size, config.window_size, "cpu"),
                persistent=False,
            )

    def forward(self, x, attn_mask=None, force_manual=False, layer_cache=None):
        B, T_new, C = x.size()

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T_new, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T_new, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T_new, self.n_head, self.d_head).transpose(1, 2)

        if layer_cache is not None:
            k = torch.cat([layer_cache[0], k], dim=2)
            v = torch.cat([layer_cache[1], v], dim=2)
        new_cache = (k, v)
        T_total = k.size(2)

        use_full = self.is_global or self.mode == "global_baseline"

        if use_full and attn_mask is None and T_new == T_total:
            # fast path: FlashAttention
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            if use_full:
                keep = torch.tril(torch.ones(T_total, T_total, dtype=torch.bool,
                                             device=x.device))
                keep = keep[T_total - T_new:T_total].view(1, 1, T_new, T_total)
            else:
                full = self.window_mask.to(x.device)
                keep = full[:, :, T_total - T_new:T_total, :T_total]

            if attn_mask is not None:
                pad = attn_mask.bool()
                if pad.size(1) < T_total:
                    extra = torch.ones(B, T_total - pad.size(1),
                                       dtype=torch.bool, device=pad.device)
                    pad = torch.cat([pad, extra], dim=1)
                keep = keep & pad.view(B, 1, 1, T_total)

            if force_manual:
                att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
                att = att.masked_fill(~keep, float("-inf"))
                att = self.attn_dropout(F.softmax(att, dim=-1))
                y = att @ v
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=keep, is_causal=False,
                    dropout_p=self.dropout if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.resid_dropout(self.c_proj(y)), new_cache


class EfficientBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = WindowedSelfAttention(config, layer_idx)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x, attn_mask=None, force_manual_attn=False, layer_cache=None):
        a, new_cache = self.attn(self.norm1(x), attn_mask, force_manual_attn,
                                 layer_cache)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class EfficientGPT(GPT):
    """Inherits forward, generate, _init_weights, configure_optimizers and
    num_params from GPT so the arms are identical except for attention."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_emb=nn.Embedding(config.vocab_size, config.d_model),
            pos_emb=nn.Embedding(config.block_size, config.d_model),
            h=nn.ModuleList([EfficientBlock(config, i)
                             for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.d_model),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.tok_emb.weight

        self.apply(self._init_weights)
        # same residual rescaling as the baseline — omitting it would be a
        # silent initialization difference and break the matched comparison
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))