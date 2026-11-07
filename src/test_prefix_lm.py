import torch
from prefix_lm import (PrefixModelConfig, SelfAttention, build_prefix_lm_mask,
                       suffix_loss_mask, compute_suffix_loss,
                       sample_split_indices, SPLIT_LO, SPLIT_HI)

TOL = 1e-5


def test_mask():
    print("--- 1. mask construction (T=8, split=4) ---")
    split = torch.tensor([4])
    mask = build_prefix_lm_mask(1, 8, split, "cpu")

    expected = torch.zeros((8, 8), dtype=torch.bool)
    expected[:4, :4] = True
    expected[4:, :4] = True
    expected[4:, 4:] = torch.tril(torch.ones(4, 4, dtype=torch.bool))

    assert torch.equal(mask[0, 0], expected), "prefix-LM mask is wrong"
    for row in expected.long().tolist():
        print("   ", row)
    print("  matches hand-computed matrix\n")


def test_degenerate():
    print("--- 2. degenerate splits ---")
    m0 = build_prefix_lm_mask(1, 8, torch.tensor([0]), "cpu")[0, 0]
    assert torch.equal(m0, torch.tril(torch.ones(8, 8, dtype=torch.bool))), \
        "split=0 should reduce to a pure causal LM"
    print("  split=0  -> pure causal mask")

    m8 = build_prefix_lm_mask(1, 8, torch.tensor([8]), "cpu")[0, 0]
    assert m8.all(), "split=T should be fully bidirectional"
    print("  split=T  -> fully bidirectional")

    lm = suffix_loss_mask(1, 8, torch.tensor([8]), "cpu")
    print(f"  split=T  -> {int(lm.sum())} supervised position(s) "
          f"(index 7 predicts nothing, so this is the floor)\n")


def test_isolation(device, dtype):
    print(f"--- 3. isolation ({device}, {dtype}) ---")
    torch.manual_seed(0)
    cfg = PrefixModelConfig(block_size=8, d_model=64, n_head=4, dropout=0.0)
    attn = SelfAttention(cfg).to(device=device, dtype=dtype).eval()

    split = torch.tensor([4], device=device)
    mask = build_prefix_lm_mask(1, 8, split, device)
    x = torch.randn(1, 8, cfg.d_model, device=device, dtype=dtype)

    for label, manual in (("manual", True), ("sdpa", False)):
        with torch.no_grad():
            base = attn(x, attn_mask=mask, force_manual=manual)

            xa = x.clone()
            xa[0, 3, :] = torch.randn(cfg.d_model, device=device, dtype=dtype)
            a = attn(xa, attn_mask=mask, force_manual=manual)
            d_bidir = (base[0, 1] - a[0, 1]).abs().max().item()

            xb = x.clone()
            xb[0, 7, :] = torch.randn(cfg.d_model, device=device, dtype=dtype)
            b = attn(xb, attn_mask=mask, force_manual=manual)
            d_causal = (base[0, 5] - b[0, 5]).abs().max().item()

            d_leak = (base[0, 1] - b[0, 1]).abs().max().item()

        print(f"  {label:7s} prefix bidirectional {d_bidir:.2e} | "
              f"suffix causal {d_causal:.2e} | prefix isolated {d_leak:.2e}")
        assert d_bidir > TOL, f"{label}: prefix is not bidirectional"
        assert d_causal < TOL, f"{label}: suffix causality violated"
        assert d_leak < TOL, f"{label}: suffix leaked into the prefix"
    print()


def test_loss_mask():
    print("--- 4. supervised positions ---")
    split = torch.tensor([4])
    m = suffix_loss_mask(1, 8, split, "cpu")[0]
    got = [i for i, v in enumerate(m.tolist()) if v]
    expected = [3, 4, 5, 6]
    print(f"  split=4, T=8 -> supervised {got}, expected {expected}")
    print("  (index 3 = last prefix token, predicts the first suffix token)")
    assert got == expected, "supervised positions are misaligned"

    torch.manual_seed(0)
    logits = torch.randn(1, 8, 50)
    targets = torch.randint(0, 50, (1, 8))
    l1 = compute_suffix_loss(logits, targets, split).item()

    logits2 = logits.clone()
    logits2[0, 0] = torch.randn(50)
    logits2[0, 7] = torch.randn(50)
    l2 = compute_suffix_loss(logits2, targets, split).item()
    print(f"  loss {l1:.6f} -> {l2:.6f} after perturbing unsupervised logits")
    assert abs(l1 - l2) < 1e-9, "unsupervised positions are contributing to the loss"

    logits3 = logits.clone()
    logits3[0, 5] = torch.randn(50)
    l3 = compute_suffix_loss(logits3, targets, split).item()
    assert abs(l1 - l3) > 1e-6, "supervised position does not affect the loss"
    print(f"  loss {l1:.6f} -> {l3:.6f} after perturbing a supervised logit\n")


def test_split_sampler():
    print("--- 5. split sampler ---")
    T = 1024
    g = torch.Generator().manual_seed(0)
    s = sample_split_indices(10000, T, "cpu", generator=g).float()
    frac = (s / T)
    print(f"  split/T: min {frac.min():.3f} mean {frac.mean():.3f} "
          f"max {frac.max():.3f}  (target {SPLIT_LO}-{SPLIT_HI})")
    assert frac.min() >= SPLIT_LO - 2/T and frac.max() <= SPLIT_HI + 2/T

    sup = ((T - 1) - (s - 1)) / T
    print(f"  supervised fraction of positions: mean {sup.mean():.3f}")
    print(f"  -> ~{1/sup.mean():.1f}x less supervision per token than a "
          f"causal decoder\n")


if __name__ == "__main__":
    test_mask()
    test_degenerate()
    test_isolation("cpu", torch.float32)
    if torch.cuda.is_available():
        test_isolation("cuda", torch.bfloat16)
    test_loss_mask()
    test_split_sampler()
    print("all checks passed")