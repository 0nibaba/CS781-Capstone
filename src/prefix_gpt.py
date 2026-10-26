"""
Prefix-LM model. Deliberately shares initialization and optimizer
construction with the causal decoder (imported, not copied) so that the
architecture comparison is controlled: the arms differ in attention
pattern and objective, not in how weights are set up or optimized.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from decoder_only_transformer import GPT
from prefix_lm import (PrefixModelConfig, Block, RMSNorm,
                       build_prefix_lm_mask, compute_suffix_loss,
                       sample_split_indices, SPLIT_EVAL)
from doc_mask import build_document_mask


class PrefixGPT(nn.Module):
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

    # shared with the decoder so the arms are initialized and optimized identically
    _init_weights = GPT._init_weights
    configure_optimizers = GPT.configure_optimizers
    num_params = GPT.num_params

    def forward(self, idx, targets=None, attn_mask=None, split_indices=None, return_hidden = False):
        """
        attn_mask: optional (B, T) padding mask.
        split_indices: optional (B,) explicit splits; sampled if absent.
        Signature matches GPT.forward so the training loop can call either.
        """
        B, T = idx.size()
        device = idx.device
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"

        if split_indices is None:
            if self.training:
                split_indices = sample_split_indices(B, T, device)
            else:
                # fixed split for validation: mid-range of the training
                # distribution, so val measures a regime the model trains on
                split_indices = torch.full((B,), int(SPLIT_EVAL * T),
                                           dtype=torch.long, device=device)

        prefix_mask = build_prefix_lm_mask(B, T, split_indices, device,
                                           pad_mask=attn_mask)
        if getattr(self.config, "doc_mask", False):
            # conjunction: a position may attend only within its own document
            # AND within the prefix-LM pattern
            prefix_mask = prefix_mask & build_document_mask(
                idx, self.config.eos_id)

        # a fully-masked row would make softmax produce NaN. the diagonal is
        # always permitted by both masks (same document, causal), so this
        # cannot happen — asserted once rather than assumed.
        if getattr(self.config, "doc_mask", False):
            prefix_mask = prefix_mask & build_document_mask(
                idx, self.config.eos_id)

        # left-padded positions have no valid keys (their causal row covers
        # only other padding), which would give NaN from the softmax. allow
        # the diagonal so every query attends to at least itself; these
        # positions' outputs are discarded during generation anyway.
        eye = torch.eye(T, dtype=torch.bool, device=device).view(1, 1, T, T)
        prefix_mask = prefix_mask | eye

        pos = torch.arange(T, dtype=torch.long, device=device)
        x = self.transformer.tok_emb(idx) + self.transformer.pos_emb(pos)

        for block in self.transformer.h:
            x = block(x, attn_mask=prefix_mask)

        x = self.transformer.ln_f(x)
        
        if return_hidden:
            return x

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = compute_suffix_loss(logits, targets, split_indices)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        The prompt is the bidirectional prefix; generated tokens extend the
        causal suffix. No KV cache: the (B,1,T,T) mask is rebuilt每 step, so
        this is O(n^2) and slow. Acceptable for tests; needs a cache before
        1000-prompt generation runs.
        """
        B, prompt_len = idx.size()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            # the prefix is the original prompt, however much of it is in view
            split = torch.full((B,), min(prompt_len, idx_cond.size(1)),
                               dtype=torch.long, device=idx.device)
            logits, _ = self(idx_cond, split_indices=split)
            logits = logits[:, -1, :].clone() / max(temperature, 1e-8)

            if top_k is not None:
                v = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


# ---------------- checks ----------------

def check_params():
    from decoder_only_transformer import ModelConfig
    print("--- parameter count vs the causal decoder ---")
    dec = GPT(ModelConfig())
    pre = PrefixGPT(PrefixModelConfig())
    nd = sum(p.numel() for p in dec.parameters())
    np_ = sum(p.numel() for p in pre.parameters())
    print(f"  decoder   {nd:,}")
    print(f"  prefix-LM {np_:,}")
    print(f"  match: {nd == np_}")
    assert nd == np_, "parameter counts differ; the comparison is not controlled"
    print(f"  transformer only (both): {pre.num_params():,}\n")


def check_initial_loss():
    print("--- initial loss over supervised positions ---")
    torch.manual_seed(0)
    cfg = PrefixModelConfig()
    m = PrefixGPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    tgt = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        _, loss = m(idx, targets=tgt)
    exp = math.log(cfg.vocab_size)
    print(f"  loss {loss.item():.4f}  expected ln({cfg.vocab_size}) = {exp:.4f}")
    print(f"  within 10.3-10.5: {10.3 <= loss.item() <= 10.5}\n")


def check_overfit(tokenizer_path, steps=800, lr=1e-3):
    """Same sanity check as the decoder, but with ~5x less supervision per
    step, so more steps are needed. The gap is itself a datapoint."""
    from tokenizers import Tokenizer
    print("--- overfit test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tok = Tokenizer.from_file(tokenizer_path)
    text = ("The High Court in revision overturned the Magistrate, stating that "
            "he acted beyond the scope of his authority. The primary question "
            "before this Court is whether the Magistrate erred in taking "
            "cognizance based on a police report filed under Section 173(2) of "
            "the Code. It is well settled that the power to direct an "
            "investigation under Section 156(3) may be exercised only at the "
            "pre-cognizance stage, and not thereafter.")
    ids = tok.encode(text).ids
    print(f"  paragraph -> {len(ids)} tokens")

    cfg = PrefixModelConfig(vocab_size=tok.get_vocab_size(), block_size=256,
                            n_layer=2, n_head=4, d_model=128, dropout=0.0)
    model = PrefixGPT(cfg).to(device)
    opt = model.configure_optimizers(0.0, lr, device)

    data = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    x, y = data[:, :-1], data[:, 1:]

    model.train()
    for s in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        if s % 100 == 0 or s == steps:
            print(f"  step {s:4d} | loss {loss.item():.4f}")

    final = loss.item()
    print(f"\n  {'PASS' if final < 0.2 else 'CHECK'} final loss {final:.4f}")
    if final >= 0.2:
        print("  note: ~5x less supervision per step than the decoder, so more "
              "steps or a higher lr may be needed before concluding a bug")


if __name__ == "__main__":
    import sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    check_params()
    check_initial_loss()
    check_overfit(os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json"))