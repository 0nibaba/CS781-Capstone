import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

def get_doc_ids(idx, eos_id):
    is_eos = (idx == eos_id).long()
    shifted = torch.cat([torch.zeros_like(is_eos[:, :1]), is_eos[:, :-1]], dim=1)
    return shifted.cumsum(dim=-1)


def build_document_mask(idx, eos_id, pad_mask=None):
    doc_ids = get_doc_ids(idx, eos_id)
    B, T = doc_ids.size()
    same_doc = doc_ids.view(B, 1, T, 1) == doc_ids.view(B, 1, 1, T)
    if pad_mask is not None:
        same_doc = same_doc & pad_mask.bool().view(B, 1, 1, T)
    return same_doc

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = False


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

        keep = self.causal_mask[:, :, T_total - T_new:T_total, :T_total]

        if attn_mask is not None:
            if attn_mask.dim() == 2:                  # (B, T) padding
                pad = attn_mask.bool()
                if pad.size(1) < T_total:             # extend over cached tokens
                    extra = torch.ones(B, T_total - pad.size(1),
                                       dtype=torch.bool, device=pad.device)
                    pad = torch.cat([pad, extra], dim=1)
                keep = keep & pad.view(B, 1, 1, T_total)
            elif attn_mask.dim() == 4:                # (B, 1, T, T) document
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


TOL = 1e-5


def test_doc_ids():
    print("--- stage 1: document ids ---")
    eos = 2
    cases = [
        ([10, 11, eos, 12, 13, eos, 14, 15],
         [0, 0, 0, 1, 1, 1, 2, 2],
         "EOS at 2 and 5"),
        ([10, 11, 12, eos, 13, 14, 15, 16],
         [0, 0, 0, 0, 1, 1, 1, 1],
         "EOS at 3"),
        ([eos, 10, 11, 12, 13, 14, 15, 16],
         [0, 1, 1, 1, 1, 1, 1, 1],
         "EOS at position 0"),
        ([10, 11, 12, 13, 14, 15, 16, eos],
         [0, 0, 0, 0, 0, 0, 0, 0],
         "EOS only at the end"),
    ]
    for tokens, expected, desc in cases:
        idx = torch.tensor([tokens], dtype=torch.long)
        got = get_doc_ids(idx, eos)[0].tolist()
        ok = got == expected
        print(f"  {desc:22s} {got}  {'ok' if ok else 'FAIL expected ' + str(expected)}")
        assert ok
    print("  EOS stays with the document it terminates: confirmed\n")


def test_mask():
    print("--- stage 2: mask vs hand-computed ---")
    eos = 2
    idx = torch.tensor([[10, 11, 12, eos, 13, 14, 15, 16]], dtype=torch.long)

    expected = torch.zeros((8, 8), dtype=torch.bool)
    expected[:4, :4] = True
    expected[4:, 4:] = True

    mask = build_document_mask(idx, eos)
    assert torch.equal(mask[0, 0], expected), "document mask is wrong"
    print("  same-document mask matches hand-computed matrix")

    pad = torch.ones(1, 8, dtype=torch.long)
    pad[0, :2] = 0
    mask_p = build_document_mask(idx, eos, pad_mask=pad)
    expected_p = expected.clone()
    expected_p[:, :2] = False
    assert torch.equal(mask_p[0, 0], expected_p), "padding not folded correctly"
    print("  padding mask folds in correctly")

    causal = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    print("\n  causal & same_doc (1 = attend):")
    for row in (causal & expected).long().tolist():
        print("   ", row)
    print()


def test_isolation(device, dtype):
    print(f"--- stage 3: isolation ({device}, {dtype}) ---")
    torch.manual_seed(0)
    eos = 2
    cfg = ModelConfig(block_size=16, d_model=64, n_head=4, dropout=0.0)
    attn = CausalSelfAttention(cfg).to(device=device, dtype=dtype).eval()

    idx = torch.tensor([[10, 11, 12, eos, 13, 14, 15, 16]],
                       dtype=torch.long, device=device)
    pad = torch.ones(1, 8, dtype=torch.long, device=device)
    doc_mask = build_document_mask(idx, eos, pad_mask=pad)

    x = torch.randn(1, 8, cfg.d_model, device=device, dtype=dtype)

    for label, manual in (("manual", True), ("sdpa", False)):
        with torch.no_grad():
            a, _ = attn(x, attn_mask=doc_mask, force_manual=manual)
            xp = x.clone()
            xp[0, 1, :] = torch.randn(cfg.d_model, device=device, dtype=dtype)
            b, _ = attn(xp, attn_mask=doc_mask, force_manual=manual)

        d1 = (a[0, :4] - b[0, :4]).abs().max().item()
        d2 = (a[0, 4:] - b[0, 4:]).abs().max().item()
        print(f"  {label:7s} doc1 changed {d1:.2e} | doc2 changed {d2:.2e}")
        assert d1 > TOL, f"{label}: perturbation did not affect its own document"
        assert d2 < TOL, f"{label}: LEAK across document boundary ({d2:.2e})"

    print("  no cross-document leakage\n")


def test_throughput(device):
    if device != "cuda":
        print("--- stage 4: throughput — skipped (no cuda) ---")
        return

    print("--- stage 4: throughput at real scale ---")
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = ModelConfig()
    eos = 0
    B, T = 8, cfg.block_size

    attn = CausalSelfAttention(cfg).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(B, T, cfg.d_model, device=device, dtype=torch.bfloat16)

    # realistic separator density: ~1 per 300 tokens, as measured on `high`
    idx = torch.randint(3, cfg.vocab_size, (B, T), device=device)
    idx[torch.rand(B, T, device=device) < (1 / 300)] = eos
    doc_mask = build_document_mask(idx, eos)
    n_docs = get_doc_ids(idx, eos)[:, -1].float().mean().item() + 1
    print(f"  {n_docs:.1f} documents per {T}-token block on average")

    def bench(mask, n=30):
        with torch.no_grad():
            for _ in range(5):
                attn(x, attn_mask=mask)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                attn(x, attn_mask=mask)
            torch.cuda.synchronize()
        return B * T * n / (time.time() - t0)

    base = bench(None)
    doc = bench(doc_mask)
    print(f"  is_causal (FlashAttention) : {base:>12,.0f} tok/s")
    print(f"  explicit document mask     : {doc:>12,.0f} tok/s")
    print(f"  cost of the explicit mask  : {100 * (1 - doc / base):.1f}%")
    print("  (attention only, not a full training step)\n")


if __name__ == "__main__":
    test_doc_ids()
    test_mask()

    test_isolation("cpu", torch.float32)
    if torch.cuda.is_available():
        test_isolation("cuda", torch.float32)
        test_isolation("cuda", torch.bfloat16)   # the regime training runs in
    test_throughput("cuda" if torch.cuda.is_available() else "cpu")