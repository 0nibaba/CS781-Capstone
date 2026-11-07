import torch
from prefix_lm import PrefixModelConfig, build_prefix_lm_mask
from prefix_gpt import PrefixGPT
from doc_mask import build_document_mask

TOL = 1e-5
EOS = 0


def test_combined_mask():
    print("--- 1. combined mask, hand-checked (T=12, split=8, EOS at 5) ---")
    idx = torch.tensor([[10, 11, 12, 13, 14, EOS, 15, 16, 17, 18, 19, 20]])
    split = torch.tensor([8])

    pre = build_prefix_lm_mask(1, 12, split, "cpu")
    doc = build_document_mask(idx, EOS)
    comb = (pre & doc)[0, 0]

    print("  combined (1 = attend):")
    for i, row in enumerate(comb.long().tolist()):
        print(f"    {i:2d} {row}")

    assert not comb[0, 6:].any(), "doc 0 attends into doc 1"
    assert not comb[6, :6].any(), "doc 1 attends back into doc 0"
    assert comb[0, :6].any(), "prefix row lost all its own document"
    print("\n  cross-document attention removed in both directions")

    assert comb.any(dim=-1).all(), "a row attends to nothing -> NaN softmax"
    print("  every row attends to at least one position\n")


def test_no_empty_rows():
    print("--- 2. no fully-masked rows over random realistic inputs ---")
    torch.manual_seed(0)
    T = 1024
    bad = 0
    for _ in range(200):
        idx = torch.randint(1, 32000, (4, T))
        for b in range(4):
            n = torch.randint(0, 3, (1,)).item()
            if n:
                pos = torch.randperm(T)[:n]
                idx[b, pos] = EOS
        split = torch.randint(int(0.7 * T), int(0.9 * T) + 1, (4,))
        m = build_prefix_lm_mask(4, T, split, "cpu") & build_document_mask(idx, EOS)
        if not m.any(dim=-1).all():
            bad += 1
    print(f"  {bad} of 200 batches contained an all-masked row")
    assert bad == 0, "fully-masked rows occur; softmax would return NaN"
    print("  (the diagonal is always permitted by both masks, so this is"
          " structurally guaranteed)\n")


def test_isolation():
    print("--- 3. end-to-end: document isolation through the full model ---")
    torch.manual_seed(0)
    cfg = PrefixModelConfig(block_size=32, d_model=64, n_head=4, n_layer=2,
                            dropout=0.0, doc_mask=True, eos_id=EOS)
    m = PrefixGPT(cfg).eval()

    idx = torch.randint(1, 1000, (1, 16))
    idx[0, 7] = EOS
    split = torch.tensor([12])

    with torch.no_grad():
        a, _ = m(idx, split_indices=split)
        idx2 = idx.clone()
        idx2[0, 2] = (idx2[0, 2] + 11) % 1000
        b, _ = m(idx2, split_indices=split)

    d_own = (a[0, 5] - b[0, 5]).abs().max().item()
    d_other = (a[0, 10:] - b[0, 10:]).abs().max().item()
    print(f"  same document (pos 5):  {d_own:.2e}")
    print(f"  other document (10-15): {d_other:.2e}")
    assert d_own > TOL, "perturbation had no effect in its own document"
    assert d_other < TOL, "LEAK across the document boundary"
    print("  isolation holds through 2 layers\n")


def test_bidirectionality_within_doc():
    print("--- 4. prefix is still bidirectional WITHIN a document ---")
    torch.manual_seed(0)
    cfg = PrefixModelConfig(block_size=32, d_model=64, n_head=4, n_layer=2,
                            dropout=0.0, doc_mask=True, eos_id=EOS)
    m = PrefixGPT(cfg).eval()

    idx = torch.randint(1, 1000, (1, 16))
    idx[0, 11] = EOS
    split = torch.tensor([12])

    with torch.no_grad():
        a, _ = m(idx, split_indices=split)
        idx2 = idx.clone()
        idx2[0, 9] = (idx2[0, 9] + 11) % 1000
        b, _ = m(idx2, split_indices=split)

    d_early = (a[0, 2] - b[0, 2]).abs().max().item()
    print(f"  early prefix (pos 2) moves when pos 9 changes: {d_early:.2e}")
    assert d_early > TOL, "bidirectionality lost inside the document"
    print("  document masking did not break the bidirectional prefix\n")


if __name__ == "__main__":
    test_combined_mask()
    test_no_empty_rows()
    test_isolation()
    test_bidirectionality_within_doc()
    print("all checks passed")