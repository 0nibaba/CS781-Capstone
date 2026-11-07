import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field

from decoder_only_transformer import GPT, ModelConfig, RMSNorm, FeedForward


@dataclass
class EfficientModelConfig(ModelConfig):
    window_size: int = 256
    global_layers: tuple = (3, 7, 11)
    attention_mode: str = "masked"


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
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))