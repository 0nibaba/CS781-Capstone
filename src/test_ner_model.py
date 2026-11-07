import torch
from ner_model import TokenClassifier, load_base, N_TAGS

CKPT = "../checkpoints/arch2_prefix_3B/ckpt_best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
base, arch, d = load_base(CKPT, device)
m = TokenClassifier(base, arch, d).to(device)

print(f"arch {arch}, d_model {d}, {N_TAGS} tags")
t, a = m.trainable_parameters()
print(f"trainable {t:,} / {a:,}")

ids = torch.randint(0, 32000, (2, 128), device=device)
lab = torch.randint(0, N_TAGS, (2, 128), device=device)
logits, loss = m(ids, labels=lab)
print(f"logits {tuple(logits.shape)}  loss {loss.item():.4f} "
      f"(ln({N_TAGS}) = {torch.log(torch.tensor(float(N_TAGS))):.4f})")

m.eval()
with torch.no_grad():
    h1 = m.hidden(ids)
    ids2 = ids.clone()
    ids2[0, 100] = (ids2[0, 100] + 7) % 32000
    h2 = m.hidden(ids2)
d_early = (h1[0, 20] - h2[0, 20]).abs().max().item()
print(f"\nposition 20 changes when position 100 changes: {d_early:.2e}")
print("  > 0 means bidirectional (right context visible)")
print("  = 0 means causal")