import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from doc_mask import build_document_mask


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = False
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
                 .view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, attn_mask=None, force_manual=False, layer_cache=None):
        """
        attn_mask: (B, T_prompt) — 1 for real tokens, 0 for left padding.
                   Covers the PROMPT only; cached generated tokens are always
                   real and are appended as 1s internally.
        layer_cache: (k, v) each (B, n_head, T_cached, d_head), or None.
        Returns (output, new_cache).
        """
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

        # rows T_total-T_new .. T_total, columns 0 .. T_total.
        # prefill  -> the usual triangle
        # decode   -> a single row of all-True (everything cached is in the past)
        keep = self.causal_mask[:, :, T_total - T_new:T_total, :T_total]

        if attn_mask is not None:
            if attn_mask.dim() == 2:                     # (B, T) padding mask
                pad = attn_mask.bool()
                if pad.size(1) < T_total:                # extend over cached tokens
                    extra = torch.ones(B, T_total - pad.size(1),
                                       dtype=torch.bool, device=pad.device)
                    pad = torch.cat([pad, extra], dim=1)
                keep = keep & pad.view(B, 1, 1, T_total)
            elif attn_mask.dim() == 4:                   # (B, 1, T, T) document mask
                keep = keep & attn_mask
            else:
                raise ValueError(f"attn_mask must be 2D or 4D, got {attn_mask.dim()}D")

        if self.use_flash and not force_manual:
            if attn_mask is None and T_new == T_total:
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True,
                    dropout_p=self.dropout if self.training else 0.0)
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=keep, is_causal=False,
                    dropout_p=self.dropout if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            att = att.masked_fill(~keep, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.resid_dropout(self.c_proj(y)), new_cache


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
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x, attn_mask=None, force_manual_attn=False, layer_cache=None):
        a, new_cache = self.attn(self.norm1(x), attn_mask, force_manual_attn, layer_cache)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_emb=nn.Embedding(config.vocab_size, config.d_model),
            pos_emb=nn.Embedding(config.block_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.d_model),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.tok_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.pos_emb.weight.numel()
            n -= self.transformer.tok_emb.weight.numel()
        return n

    def configure_optimizers(self, weight_decay, learning_rate, device_type,
                             betas=(0.9, 0.95)):
        import inspect
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        nodecay = [p for p in params.values() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                 fused=(fused_ok and device_type == "cuda"))

    def forward(self, idx, targets=None, attn_mask=None,
                past_caches=None, use_cache=False):
        b, t = idx.size()

        offset = 0 if past_caches is None else past_caches[0][0].size(2)
        assert offset + t <= self.config.block_size, \
            f"position {offset + t} exceeds block_size {self.config.block_size}"

        if self.config.doc_mask and past_caches is None:
            attn_mask = build_document_mask(idx, self.config.eos_id, pad_mask=attn_mask)

        pos = torch.arange(offset, offset + t, dtype=torch.long, device=idx.device)
        x = self.transformer.tok_emb(idx) + self.transformer.pos_emb(pos)

        new_caches = []
        for i, block in enumerate(self.transformer.h):
            lc = past_caches[i] if past_caches is not None else None
            x, nc = block(x, attn_mask=attn_mask, layer_cache=lc)
            new_caches.append(nc)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))

        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Simple unbatched/unpadded generation. Padded batches use the
        standalone loop in local_eval.py."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].clone() / max(temperature, 1e-8)
            if top_k is not None:
                v = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


# ---------------- checks ----------------

def run_component_checks():
    torch.manual_seed(0)
    config = ModelConfig(dropout=0.0)
    assert config.bias is False

    print("--- 1. causal self-attention ---")
    attn = CausalSelfAttention(config).eval()
    x = torch.randn(2, 16, config.d_model)

    with torch.no_grad():
        out_manual, _ = attn(x, force_manual=True)
        out_flash, _ = attn(x, force_manual=False)

    print(f"shape {tuple(x.shape)} -> {tuple(out_manual.shape)}")
    print(f"manual == flash: {torch.allclose(out_manual, out_flash, atol=1e-5)}  "
          f"(max diff {(out_manual - out_flash).abs().max():.2e})")

    x2 = x.clone()
    x2[:, 10, :] = torch.randn(config.d_model)
    with torch.no_grad():
        out2, _ = attn(x2, force_manual=True)
    before = (out_manual[:, :10] - out2[:, :10]).abs().max().item()
    after = (out_manual[:, 10:] - out2[:, 10:]).abs().max().item()
    print(f"positions 0-9 unchanged: {before < 1e-7}  (max diff {before:.1e})")
    print(f"positions 10+ changed:   {after > 1e-4}  (max diff {after:.1e})")

    m = torch.ones(2, 16, dtype=torch.long)
    m[:, :4] = 0
    xp = x.clone()
    with torch.no_grad():
        o1, _ = attn(xp, attn_mask=m, force_manual=True)
        xp[:, :4, :] = torch.randn(2, 4, config.d_model)
        o2, _ = attn(xp, attn_mask=m, force_manual=True)
    d = (o1[:, 4:] - o2[:, 4:]).abs().max().item()
    print(f"padding ignored at real positions: {d < 1e-6}  (max diff {d:.1e})")

    print("\n--- 2. feed-forward ---")
    ffn = FeedForward(config).eval()
    with torch.no_grad():
        print(f"shape {tuple(x.shape)} -> {tuple(ffn(x).shape)}")

    print("\n--- 3. block ---")
    block = Block(config).eval()
    with torch.no_grad():
        out_block, _ = block(x)
        print(f"shape {tuple(x.shape)} -> {tuple(out_block.shape)}")
        block.attn.c_proj.weight.zero_()
        block.ffn.c_proj.weight.zero_()
        out_zeroed, _ = block(x)
    print(f"residual path clean with zeroed projections: "
          f"{torch.allclose(x, out_zeroed, atol=1e-6)}")


def run_model_checks():
    print("\n--- 4. full model ---")
    config = ModelConfig(dropout=0.0)
    model = GPT(config)

    print(f"weight tying (identity): "
          f"{model.lm_head.weight is model.transformer.tok_emb.weight}")

    total = sum(p.numel() for p in model.parameters())
    emb = model.transformer.tok_emb.weight.numel()
    model.lm_head.weight = nn.Parameter(model.transformer.tok_emb.weight.clone())
    untied = sum(p.numel() for p in model.parameters())
    model.lm_head.weight = model.transformer.tok_emb.weight

    print(f"params tied:        {total:,}")
    print(f"params untied:      {untied:,}")
    print(f"saved by tying:     {untied - total:,}  (expected {emb:,})")
    print(f"transformer only:   {model.num_params():,}")

    idx = torch.randint(0, config.vocab_size, (2, 16))
    tgt = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        logits, loss = model(idx, tgt)
    print(f"\nshape {tuple(idx.shape)} -> {tuple(logits.shape)}")
    print(f"initial loss: {loss.item():.4f}  (expected "
          f"{math.log(config.vocab_size):.4f})")
    print(f"within 10.3-10.5: {10.3 <= loss.item() <= 10.5}")


def run_cache_checks():
    """The cached and uncached paths must produce IDENTICAL logits."""
    print("\n--- 5. KV cache equivalence ---")
    torch.manual_seed(0)
    cfg = ModelConfig(n_layer=4, n_head=4, d_model=256, block_size=128, dropout=0.0)
    model = GPT(cfg).eval()

    B, T_prompt, n_gen = 3, 20, 12
    idx = torch.randint(0, cfg.vocab_size, (B, T_prompt))
    mask = torch.ones(B, T_prompt, dtype=torch.long)
    mask[:, :5] = 0                       # left padding
    idx[:, :5] = 0

    # --- uncached: full forward every step, greedy ---
    seq_a = idx.clone()
    with torch.no_grad():
        for _ in range(n_gen):
            logits, _ = model(seq_a, attn_mask=torch.cat(
                [mask, torch.ones(B, seq_a.size(1) - T_prompt, dtype=torch.long)], 1))
            seq_a = torch.cat([seq_a, logits[:, -1, :].argmax(-1, keepdim=True)], 1)

    # --- cached: prefill once, then one token at a time ---
    with torch.no_grad():
        logits, _, caches = model(idx, attn_mask=mask, use_cache=True)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        seq_b = torch.cat([idx, nxt], 1)
        for _ in range(n_gen - 1):
            logits, _, caches = model(nxt, attn_mask=mask,
                                      past_caches=caches, use_cache=True)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            seq_b = torch.cat([seq_b, nxt], 1)

    same = torch.equal(seq_a, seq_b)
    print(f"cached == uncached token ids: {same}")
    if not same:
        diff = (seq_a != seq_b).nonzero()
        print(f"  FAIL — first divergence at {diff[0].tolist()}")
        print(f"  uncached: {seq_a[0, T_prompt:].tolist()}")
        print(f"  cached:   {seq_b[0, T_prompt:].tolist()}")
        print("  usual causes: position offset, or the padding mask not "
              "extended over cached generated tokens")


if __name__ == "__main__":
    run_component_checks()
    run_model_checks()
    run_cache_checks()