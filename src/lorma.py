import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LoRMAQKV(nn.Module):

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

        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for t in self.targets:
            a = torch.empty(r, d_model)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.A[t] = nn.Parameter(a)
            self.B[t] = nn.Parameter(torch.zeros(d_model, r))

    def forward(self, x):
        out = self.base(x)
        q, k, v = out.split(self.d, dim=-1)
        parts = {"q": q, "k": k, "v": v}

        for t in self.targets:
            h = parts[t]
            u = h @ self.A[t].T
            parts[t] = h + self.scaling * (u @ self.B[t].T)

        return torch.cat([parts["q"], parts["k"], parts["v"]], dim=-1)

    @torch.no_grad()
    def merged_weight(self):
        W = self.base.weight.data.clone()
        for i, t in enumerate(("q", "k", "v")):
            if t not in self.targets:
                continue
            sl = slice(i * self.d, (i + 1) * self.d)
            BA = self.B[t] @ self.A[t]
            T = self.scaling * BA + torch.eye(self.d, device=BA.device)
            W[sl] = T @ W[sl]
        return W


def apply_lorma(model, r=8, alpha=16.0, targets=("q", "v"), verbose=True):
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