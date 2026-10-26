import os
import math
import time
import torch
from efficient_attn import (EfficientModelConfig, EfficientGPT,
                            WindowedSelfAttention, build_window_mask)
from decoder_only_transformer import GPT, ModelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-5


def test_mask():
    print("--- 1. window mask (T=8, W=2) ---")
    m = build_window_mask(8, 2, "cpu")[0, 0]
    expected = torch.zeros((8, 8), dtype=torch.bool)
    for i in range(8):
        for j in range(8):
            expected[i, j] = (i >= j) and (i - j) <= 2
    assert torch.equal(m, expected), "window mask is wrong"
    for row in expected.long().tolist():
        print("   ", row)
    print("  causal band of width W+1 = 3, matches hand-computed\n")


def test_isolation(device, dtype):
    print(f"--- 2. isolation ({device}, {dtype}) ---")
    torch.manual_seed(0)
    cfg = EfficientModelConfig(block_size=16, d_model=64, n_head=4,
                               window_size=2, global_layers=(), dropout=0.0)
    attn = WindowedSelfAttention(cfg, layer_idx=0).to(device=device,
                                                      dtype=dtype).eval()
    x = torch.randn(1, 8, cfg.d_model, device=device, dtype=dtype)

    for label, manual in (("manual", True), ("sdpa", False)):
        with torch.no_grad():
            base, _ = attn(x, force_manual=manual)
            xp = x.clone()
            xp[0, 1, :] = torch.randn(cfg.d_model, device=device, dtype=dtype)
            pert, _ = attn(xp, force_manual=manual)

        inside = (base[0, 3] - pert[0, 3]).abs().max().item()    # 3-1 = 2 <= W
        outside = (base[0, 5] - pert[0, 5]).abs().max().item()   # 5-1 = 4 > W
        print(f"  {label:7s} inside window {inside:.2e} | "
              f"outside window {outside:.2e}")
        assert inside > TOL, f"{label}: in-window token had no effect"
        assert outside < TOL, f"{label}: token outside the window leaked in"
    print()


def test_params():
    print("--- 3. parameter count vs the baseline decoder ---")
    nd = sum(p.numel() for p in GPT(ModelConfig()).parameters())
    ne = sum(p.numel() for p in EfficientGPT(EfficientModelConfig()).parameters())
    print(f"  decoder   {nd:,}")
    print(f"  efficient {ne:,}")
    assert nd == ne, "parameter counts differ; windowing adds no parameters"
    print("  identical (windowing changes the mask, not the weights)\n")


def test_initial_loss():
    """Compare distributions, not one value against a theoretical number.
    Also checks whether the architectures initialize identically from the same
    seed — EfficientGPT rebuilds its modules rather than reusing GPT.__init__,
    so the sequence of random draws may differ."""
    print("--- 4. initial loss: efficient vs decoder, 5 seeds ---")
    ecfg, dcfg = EfficientModelConfig(), ModelConfig()

    def sample(make, cfg, seed):
        torch.manual_seed(seed)
        m = make(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 64))
        tgt = torch.randint(0, cfg.vocab_size, (2, 64))
        with torch.no_grad():
            _, loss = m(idx, tgt)
        del m
        return loss.item()

    eff = [sample(EfficientGPT, ecfg, s) for s in range(5)]
    dec = [sample(GPT, dcfg, s) for s in range(5)]

    print(f"  efficient {[round(x, 4) for x in eff]}")
    print(f"            mean {sum(eff)/5:.4f}  spread {max(eff)-min(eff):.4f}")
    print(f"  decoder   {[round(x, 4) for x in dec]}")
    print(f"            mean {sum(dec)/5:.4f}  spread {max(dec)-min(dec):.4f}")
    print(f"  ln({ecfg.vocab_size}) = {math.log(ecfg.vocab_size):.4f}")

    gap = abs(sum(eff)/5 - sum(dec)/5)
    pooled = (max(eff)-min(eff) + max(dec)-min(dec)) / 2
    print(f"  mean gap {gap:.4f} vs typical spread {pooled:.4f} -> "
          f"{'within noise' if gap < pooled else 'SYSTEMATIC, investigate'}")

    same = all(abs(a - b) < 1e-6 for a, b in zip(eff, dec))
    print(f"  identical per-seed init: {same}")
    if not same:
        print("    expected — EfficientGPT builds modules in a different order,")
        print("    so the same seed yields different weights. The arms are")
        print("    matched in architecture, data and budget, NOT in init.")
    print()


def test_receptive_field():
    print("--- 5. effective receptive field ---")
    cfg = EfficientModelConfig()
    g = sorted(tuple(cfg.global_layers))
    runs, prev = [], -1
    for gl in g + [cfg.n_layer]:
        runs.append(gl - prev - 1)
        prev = gl
    print(f"  window {cfg.window_size}, global layers {g}")
    print(f"  consecutive windowed runs: {runs}")
    print(f"  longest run {max(runs)} layers -> effective field "
          f"~{max(runs) * cfg.window_size} tokens before a global layer\n")


def test_throughput(device, trials=5):
    """Interleaved trials with a stated outlier rule. On a shared GPU the
    baseline alone has been observed to swing 22% between consecutive
    measurements, so single comparisons are meaningless."""
    if device != "cuda":
        print("--- 6. throughput — skipped (no cuda) ---")
        return
    print(f"--- 6. throughput ({trials} interleaved trials) ---")
    torch.backends.cuda.matmul.allow_tf32 = True
    B, T = 8, 1024

    models = {}
    for mode in ("global_baseline", "masked"):
        cfg = EfficientModelConfig(attention_mode=mode)
        models[mode] = EfficientGPT(cfg).to(device=device,
                                            dtype=torch.bfloat16).eval()
    x = torch.randint(0, EfficientModelConfig().vocab_size, (B, T), device=device)

    def bench(m, n=10):
        with torch.no_grad():
            for _ in range(3):
                m(x)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                m(x)
            torch.cuda.synchronize()
        return B * T * n / (time.time() - t0)

    full, win = [], []
    for _ in range(trials):
        full.append(bench(models["global_baseline"]))
        win.append(bench(models["masked"]))

    med_full = sorted(full)[len(full) // 2]
    keep = [(f, w) for f, w in zip(full, win) if abs(f / med_full - 1) < 0.10]
    print(f"  discarding {trials - len(keep)} trial(s) where the baseline "
          f"deviates >10% from its median (contention)")

    if not keep:
        print("  no usable trials — GPU too contended to measure\n")
        return

    for f, w in keep:
        print(f"    full {f:>9,.0f} | windowed {w:>9,.0f} | "
              f"{100*(w/f-1):+6.1f}%")

    deltas = sorted(100 * (w / f - 1) for f, w in keep)
    print(f"\n  median {deltas[len(deltas)//2]:+.1f}%  "
          f"range {deltas[0]:+.1f}% to {deltas[-1]:+.1f}%")
    print("  negative => masking saves no FLOPs and costs the FlashAttention")
    print("  fast path, so 'efficient' attention is slower than full attention\n")

    for m in models.values():
        del m
    torch.cuda.empty_cache()


def test_overfit(tokenizer_path, steps=500, lr=1e-3, window=8):
    """Catches wiring bugs in the inherited forward. The window is set small so
    the restriction actually binds — at W=256 on an 84-token paragraph it would
    be inert and this would test nothing. global_layers is empty so no layer
    escapes the window."""
    from tokenizers import Tokenizer
    print(f"--- 7. overfit test (W={window}, no global layers) ---")
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
    print(f"  paragraph -> {len(ids)} tokens, 2 layers, window {window}")
    print(f"  effective receptive field ~{2*window} tokens")

    cfg = EfficientModelConfig(vocab_size=tok.get_vocab_size(), block_size=256,
                               n_layer=2, n_head=4, d_model=128, dropout=0.0,
                               window_size=window, global_layers=())
    model = EfficientGPT(cfg).to(device)
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
    if final < 0.2:
        print(f"\n  PASS final loss {final:.4f} — wiring is sound\n")
    else:
        print(f"\n  final loss {final:.4f} — not necessarily a bug. With a")
        print(f"  ~{2*window}-token receptive field the model cannot see the")
        print("  whole paragraph, so some positions may be genuinely ambiguous.")
        print("  Re-running at window=64 to separate a wiring bug from a real")
        print("  limitation of the restricted window:\n")
        test_overfit(tokenizer_path, steps=steps, lr=lr, window=64)


if __name__ == "__main__":
    test_mask()
    test_isolation("cpu", torch.float32)
    if torch.cuda.is_available():
        test_isolation("cuda", torch.bfloat16)
    test_params()
    test_initial_loss()
    test_receptive_field()
    test_throughput("cuda" if torch.cuda.is_available() else "cpu")
    test_overfit(os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json"))