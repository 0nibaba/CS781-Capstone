import math
import torch
from encoder_decoder import (EncDecConfig, EncoderDecoder, CrossAttention)
from decoder_only_transformer import GPT, ModelConfig

TOL = 1e-5


def test_params():
    print("--- 1. parameter count ---")
    base = sum(p.numel() for p in GPT(ModelConfig()).parameters())
    ed = EncoderDecoder(EncDecConfig())
    n = sum(p.numel() for p in ed.parameters())
    print(f"  baseline decoder   {base:,}")
    print(f"  encoder-decoder    {n:,}")
    print(f"  difference         {n - base:+,}  ({100*(n-base)/base:+.4f}%)")
    assert abs(n - base) < 5000, "parameter mismatch larger than expected"
    print(f"  4 enc + 6 dec, transformer only {ed.num_params():,}\n")


def test_cross_attention_shapes():
    print("--- 2. cross-attention with T_enc != T_dec ---")
    cfg = EncDecConfig(d_model=64, n_head=4)
    ca = CrossAttention(cfg).eval()
    x = torch.randn(2, 7, 64)          # decoder
    e = torch.randn(2, 13, 64)         # encoder, different length
    with torch.no_grad():
        y = ca(x, e)
    print(f"  dec {tuple(x.shape)} x enc {tuple(e.shape)} -> {tuple(y.shape)}")
    assert y.shape == x.shape
    print("  output matches decoder shape\n")


def test_encoder_padding():
    print("--- 3. padded encoder positions are ignored ---")
    torch.manual_seed(0)
    cfg = EncDecConfig(d_model=64, n_head=4, n_encoder_layer=2,
                       n_decoder_layer=2, block_size=32)
    m = EncoderDecoder(cfg).eval()

    enc = torch.randint(1, 1000, (1, 10))
    dec = torch.randint(1, 1000, (1, 6))
    pad = torch.ones(1, 10, dtype=torch.long)
    pad[0, 7:] = 0                              # last 3 are padding

    with torch.no_grad():
        a = m.decode(dec, m.encode(enc, pad), pad)
        enc2 = enc.clone()
        enc2[0, 7:] = torch.randint(1, 1000, (3,))   # change the pads
        b = m.decode(dec, m.encode(enc2, pad), pad)

    d = (a - b).abs().max().item()
    print(f"  output change after perturbing padded encoder tokens: {d:.2e}")
    assert d < TOL, "padded encoder positions leak into the decoder"
    print("  padding correctly ignored\n")


def test_decoder_causality():
    print("--- 4. decoder self-attention is causal ---")
    torch.manual_seed(0)
    cfg = EncDecConfig(d_model=64, n_head=4, n_encoder_layer=2,
                       n_decoder_layer=2, block_size=32)
    m = EncoderDecoder(cfg).eval()
    enc = torch.randint(1, 1000, (1, 10))
    dec = torch.randint(1, 1000, (1, 8))

    with torch.no_grad():
        eo = m.encode(enc)
        a = m.decode(dec, eo)
        dec2 = dec.clone()
        dec2[0, 6] = (dec2[0, 6] + 7) % 1000     # change a LATE decoder token
        b = m.decode(dec2, eo)

    early = (a[0, :6] - b[0, :6]).abs().max().item()
    late = (a[0, 6:] - b[0, 6:]).abs().max().item()
    print(f"  positions 0-5 (before the change): {early:.2e}")
    print(f"  positions 6+  (at/after):          {late:.2e}")
    assert early < TOL, "decoder is not causal"
    assert late > TOL, "change had no effect at all"
    print("  causality holds\n")


def test_initial_loss():
    print("--- 5. initial loss ---")
    torch.manual_seed(0)
    cfg = EncDecConfig()
    m = EncoderDecoder(cfg).eval()
    enc = torch.randint(0, cfg.vocab_size, (2, 128))
    dec = torch.randint(0, cfg.vocab_size, (2, 64))
    tgt = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        _, loss = m(enc, dec, tgt)
    print(f"  {loss.item():.4f} vs ln({cfg.vocab_size}) = "
          f"{math.log(cfg.vocab_size):.4f}")
    print(f"  within 10.3-10.6: {10.3 <= loss.item() <= 10.6}\n")


def test_overfit(tokenizer_path, steps=600, lr=1e-3):
    """Memorise one corrupted paragraph. Catches wiring bugs across both
    stacks and the cross-attention."""
    import random
    from tokenizers import Tokenizer
    from span_corruption import corrupt_sequence

    print("--- 6. overfit test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = random.Random(0)

    tok = Tokenizer.from_file(tokenizer_path)
    sent = [tok.token_to_id(f"<|unused{i}|>") for i in range(10)]
    text = ("The High Court in revision overturned the Magistrate, stating that "
            "he acted beyond the scope of his authority. The primary question "
            "before this Court is whether the Magistrate erred in taking "
            "cognizance based on a police report filed under Section 173(2) of "
            "the Code.")
    ids = tok.encode(text).ids
    enc, din, dt = corrupt_sequence(ids, sent, p=0.15, mean_span=5, rng=rng)
    print(f"  {len(ids)} tokens -> enc {len(enc)}, dec {len(dt)}")

    cfg = EncDecConfig(vocab_size=tok.get_vocab_size(), block_size=256,
                       n_encoder_layer=2, n_decoder_layer=2, n_head=4,
                       d_model=128, dropout=0.0)
    model = EncoderDecoder(cfg).to(device)
    opt = model.configure_optimizers(0.0, lr, device)

    e = torch.tensor([enc], dtype=torch.long, device=device)
    di = torch.tensor([din], dtype=torch.long, device=device)
    dtg = torch.tensor([dt], dtype=torch.long, device=device)

    model.train()
    for s in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        _, loss = model(e, di, dtg)
        loss.backward()
        opt.step()
        if s % 100 == 0 or s == steps:
            print(f"  step {s:4d} | loss {loss.item():.4f}")

    final = loss.item()
    print(f"\n  {'PASS' if final < 0.2 else 'FAIL'} final loss {final:.4f}\n")


if __name__ == "__main__":
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_params()
    test_cross_attention_shapes()
    test_encoder_padding()
    test_decoder_causality()
    test_initial_loss()
    test_overfit(os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json"))