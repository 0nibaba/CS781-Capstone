"""
LoRMA+ : Low-Rank Multiplicative Adaptation.

LoRA is additive:        h = (W0 + (a/r) B A) x
LoRMA is multiplicative: h = I(B A) W0 x

With additive inflation, I(BA) = (a/r) BA + I_d, so the transform starts as
the identity (B = 0) and the adapted layer is initially a no-op. Computed as

    h = W0 x + (a/r) B (A (W0 x))

which never forms the d x d matrix and costs the same as LoRA.

FUSED QKV. The paper targets the query and value projections separately, but
this codebase uses one fused c_attn: Linear(d, 3d) whose output is split into
Q, K, V. Applying one transform to the fused output would adapt Q, K and V
jointly with a shared (3d x 3d) transform — a different method. So the output
is sliced and separate transforms are applied to the Q and V thirds, leaving
K untouched, which matches the paper.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LoRMAQKV(nn.Module):
    """Wraps a fused QKV Linear(d, 3d), adapting only the Q and V slices."""

    def __init__(self, base: nn.Linear, d_model: int, r: int = 8,
                 alpha: float = 16.0, targets=("q", "v")):
        super().__init__()
        assert base.out_features == 3 * d_model, \
            f"expected a fused QKV projection, got {base.out_features}"
        self.base = base
        self.d = d_model
        self.r = r
        self.scaling = alpha / r
        self.targets = tuple(targets)

        for p in self.base.parameters():
            p.requires_grad = False

        # one (B, A) pair per adapted slice. B = 0 so the transform starts as
        # the identity and the model is unchanged at step 0.
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for t in self.targets:
            a = torch.empty(r, d_model)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.A[t] = nn.Parameter(a)
            self.B[t] = nn.Parameter(torch.zeros(d_model, r))

    def forward(self, x):
        out = self.base(x)                      # (..., 3d)
        q, k, v = out.split(self.d, dim=-1)
        parts = {"q": q, "k": k, "v": v}

        for t in self.targets:
            h = parts[t]
            # h + (a/r) B (A h), associativity keeping everything (..., r)
            u = h @ self.A[t].T                 # (..., r)
            parts[t] = h + self.scaling * (u @ self.B[t].T)

        return torch.cat([parts["q"], parts["k"], parts["v"]], dim=-1)

    @torch.no_grad()
    def merged_weight(self):
        """W_tuned = I(BA) W0, for zero-latency inference. Returned, not
        applied, so the original is never destroyed."""
        W = self.base.weight.data.clone()       # (3d, d_in)
        for i, t in enumerate(("q", "k", "v")):
            if t not in self.targets:
                continue
            sl = slice(i * self.d, (i + 1) * self.d)
            BA = self.B[t] @ self.A[t]
            T = self.scaling * BA + torch.eye(self.d, device=BA.device)
            W[sl] = T @ W[sl]
        return W


def apply_lorma(model, r=8, alpha=16.0, targets=("q", "v"), verbose=True):
    """
    Replace every fused QKV projection with a LoRMA-adapted one and freeze
    everything except the adapters (and the classification head, which is
    new and must train).
    """
    replaced = []

    def walk(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if name == "c_attn" and isinstance(child, nn.Linear):
                d = child.in_features
                setattr(module, name,
                        LoRMAQKV(child, d, r=r, alpha=alpha, targets=targets))
                replaced.append(path)
            else:
                walk(child, path)

    walk(model)

    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if ".A." in n or ".B." in n or n.startswith("head."):
            p.requires_grad = True

    if verbose:
        t = sum(p.numel() for p in model.parameters() if p.requires_grad)
        a = sum(p.numel() for p in model.parameters())
        print(f"  LoRMA+ r={r} alpha={alpha} targets={targets}")
        print(f"  adapted {len(replaced)} projections")
        print(f"  trainable {t:,} / {a:,} ({100*t/a:.3f}%)")
    return model


# ---------------- tests ----------------

if __name__ == "__main__":
    torch.manual_seed(0)
    d, r = 64, 4

    print("--- 1. the adapter starts as the identity ---")
    base = nn.Linear(d, 3 * d, bias=False)
    layer = LoRMAQKV(base, d, r=r)
    x = torch.randn(2, 5, d)
    before = base(x)
    after = layer(x)
    diff = (before - after).abs().max().item()
    print(f"  max |base - adapted| at init: {diff:.2e}")
    assert diff < 1e-6, "B must be zero so the layer begins as a no-op"

    print("\n--- 2. only Q and V change once B is non-zero ---")
    with torch.no_grad():
        for t in layer.targets:
            layer.B[t].normal_(0, 0.1)
    after = layer(x)
    for i, t in enumerate(("q", "k", "v")):
        sl = slice(i * d, (i + 1) * d)
        dd = (before[..., sl] - after[..., sl]).abs().max().item()
        want = "changed" if t in layer.targets else "unchanged"
        print(f"  {t}: {dd:.2e}  (expected {want})")
        if t in layer.targets:
            assert dd > 1e-6
        else:
            assert dd < 1e-9

    print("\n--- 3. merged weights reproduce the adapted forward pass ---")
    W = layer.merged_weight()
    merged = F.linear(x, W, None)
    dd = (merged - after).abs().max().item()
    print(f"  max |merged - adapted|: {dd:.2e}")
    assert dd < 1e-4, "merge does not reproduce the forward pass"

    print("\n--- 4. gradients reach the adapters and nothing else ---")
    base2 = nn.Linear(d, 3 * d, bias=False)
    l2 = LoRMAQKV(base2, d, r=r)
    for p in l2.base.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for t in l2.targets:
            l2.B[t].normal_(0, 0.1)
    l2(x).sum().backward()
    print(f"  base weight grad: {l2.base.weight.grad}")
    for t in l2.targets:
        print(f"  A[{t}] grad norm {l2.A[t].grad.norm():.4f}, "
              f"B[{t}] grad norm {l2.B[t].grad.norm():.4f}")
    assert l2.base.weight.grad is None
    assert all(l2.A[t].grad is not None for t in l2.targets)

    print("\n--- 5. parameter budget on the real model ---")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ner_model import TokenClassifier, load_base
    ck = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "checkpoints/arch2_prefix_3B/ckpt_best.pt")
    if os.path.exists(ck):
        b, arch, dm = load_base(ck, "cpu")
        m = TokenClassifier(b, arch, dm, use_crf=True)
        apply_lorma(m, r=8, alpha=16.0)
    else:
        print("  (checkpoint not found, skipped)")