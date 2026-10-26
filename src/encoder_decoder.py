"""
Architecture 4: encoder-decoder with cross-attention.

4 encoder + 6 decoder layers = 110,318,592 params, +2,304 (0.002%) over the
baseline decoder. The marginally closer 8+3 split was rejected: Phase 1 scores
generation and the decoder is what generates, so a 3-layer decoder would
compete against 12 layers of generative depth in the other architectures.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from decoder_only_transformer import GPT, RMSNorm, FeedForward
from prefix_lm import SelfAttention


@dataclass
class EncDecConfig:
    vocab_size: int = 32000
    block_size: int = 1024
    n_encoder_layer: int = 4
    n_decoder_layer: int = 6
    n_head: int = 12
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = False
    use_internal_causal_mask: bool = False   # masks are always supplied
    # kept so shared code paths that read n_layer still work
    n_layer: int = 10


class CrossAttention(nn.Module):
    """Queries from the decoder, keys/values from the encoder output. The
    attention matrix is (B, h, T_dec, T_enc) — not square."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.kv_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, enc_out, enc_pad_mask=None):
        B, T_dec, C = x.size()
        T_enc = enc_out.size(1)

        q = self.q_proj(x).view(B, T_dec, self.n_head, self.d_head).transpose(1, 2)
        k, v = self.kv_proj(enc_out).split(self.d_model, dim=2)
        k = k.view(B, T_enc, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T_enc, self.n_head, self.d_head).transpose(1, 2)

        # padded encoder positions must not be attended to
        mask = None
        if enc_pad_mask is not None:
            mask = enc_pad_mask.bool().view(B, 1, 1, T_enc).expand(B, 1, T_dec, T_enc)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T_dec, C)
        return self.resid_dropout(self.c_proj(y))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config)      # no internal causal mask
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x, self_mask):
        x = x + self.attn(self.norm1(x), self_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = SelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.cross_attn = CrossAttention(config)
        self.norm3 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x, enc_out, self_mask, enc_pad_mask):
        x = x + self.self_attn(self.norm1(x), self_mask)
        x = x + self.cross_attn(self.norm2(x), enc_out, enc_pad_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)

        self.encoder = nn.ModuleList([EncoderBlock(config)
                                      for _ in range(config.n_encoder_layer)])
        self.enc_ln_f = RMSNorm(config.d_model)

        self.decoder = nn.ModuleList([DecoderBlock(config)
                                      for _ in range(config.n_decoder_layer)])
        self.dec_ln_f = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # tied

        self.apply(self._init_weights)
        n_total = config.n_encoder_layer + config.n_decoder_layer
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * n_total))

    _init_weights = GPT._init_weights
    configure_optimizers = GPT.configure_optimizers

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    def encode(self, enc_idx, enc_pad_mask=None):
        B, T = enc_idx.size()
        pos = torch.arange(T, dtype=torch.long, device=enc_idx.device)
        x = self.tok_emb(enc_idx) + self.pos_emb(pos)

        # bidirectional: all positions visible except padding
        m = torch.ones(B, 1, T, T, dtype=torch.bool, device=enc_idx.device)
        if enc_pad_mask is not None:
            m = m & enc_pad_mask.bool().view(B, 1, 1, T)

        for blk in self.encoder:
            x = blk(x, m)
        return self.enc_ln_f(x)

    def decode(self, dec_idx, enc_out, enc_pad_mask=None):
        B, T = dec_idx.size()
        pos = torch.arange(T, dtype=torch.long, device=dec_idx.device)
        x = self.tok_emb(dec_idx) + self.pos_emb(pos)

        causal = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                       device=dec_idx.device)).view(1, 1, T, T)
        causal = causal.expand(B, 1, T, T)

        for blk in self.decoder:
            x = blk(x, enc_out, causal, enc_pad_mask)
        return self.lm_head(self.dec_ln_f(x))

    def forward(self, enc_idx, dec_idx, targets=None,
                enc_pad_mask=None, dec_pad_mask=None):
        enc_out = self.encode(enc_idx, enc_pad_mask)
        logits = self.decode(dec_idx, enc_out, enc_pad_mask)

        loss = None
        if targets is not None:
            if dec_pad_mask is not None:
                # ignore padded target positions
                targets = targets.masked_fill(~dec_pad_mask.bool(), -100)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss