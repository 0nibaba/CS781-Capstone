import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
LOGS = os.path.join(ROOT, "logs")

ARM_A = "arm_a_baseline"
ARM_B = "arm_b_docmask"
SOURCES = ["acts", "high", "supreme", "tribunal"]
SMOOTH = 32

DENSITY = {"high": 1.99, "tribunal": 1.49, "supreme": 1.22, "acts": 1.12}


def smooth(x, w=SMOOTH):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def matched_steps():
    def steps(run):
        d = os.path.join(CKPT, run)
        return {int(f[9:15]) for f in os.listdir(d)
                if f.startswith("pos_loss_") and f.endswith(".npz")}
    common = sorted(steps(ARM_A) & steps(ARM_B))
    return common


def load(run, step):
    return np.load(os.path.join(CKPT, run, f"pos_loss_{step:06d}.npz"))


def trend(delta):
    """Slope of delta vs position. Negative => masking improves with depth."""
    x = np.arange(len(delta))
    return np.polyfit(x, delta, 1)[0] * len(delta)


def main():
    common = matched_steps()
    if not common:
        print("no matched steps between the two arms")
        return

    print(f"matched eval steps: {common[0]} .. {common[-1]} "
          f"({len(common)} points)\n")

    step = common[-1]
    a, b = load(ARM_A, step), load(ARM_B, step)
    print(f"=== detailed analysis at step {step} ===\n")

    d = b["overall"] - a["overall"]
    T = len(d)
    print(f"{'':<10} {'arm A':>8} {'arm B':>8} {'delta':>8} "
          f"{'first 64':>9} {'last 64':>9} {'slope':>8} {'density':>8}")
    print("-" * 78)
    print(f"{'overall':<10} {a['overall'].mean():>8.4f} {b['overall'].mean():>8.4f} "
          f"{d.mean():>+8.4f} {d[:64].mean():>+9.4f} {d[-64:].mean():>+9.4f} "
          f"{trend(d):>+8.4f}")

    for src in SOURCES:
        ka, kb = a[f"src_{src}"], b[f"src_{src}"]
        ds = kb - ka
        print(f"{src:<10} {ka.mean():>8.4f} {kb.mean():>8.4f} "
              f"{ds.mean():>+8.4f} {ds[:64].mean():>+9.4f} {ds[-64:].mean():>+9.4f} "
              f"{trend(ds):>+8.4f} {DENSITY[src]:>8.2f}")

    print("\npositive delta = masking is WORSE")
    print("negative slope = masking improves with position (supports P2)")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(T - SMOOTH + 1) + SMOOTH // 2
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(x, smooth(d), lw=2, color="black", label="overall")
    for src in SOURCES:
        ds = b[f"src_{src}"] - a[f"src_{src}"]
        ax.plot(x, smooth(ds), lw=1.2,
                label=f"{src} ({DENSITY[src]:.2f} docs/win)")
    ax.set_xlabel("position within 1024-token block")
    ax.set_ylabel("val loss delta  (doc-mask − baseline)")
    ax.set_title(f"Effect of document masking by position — step {step}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(LOGS, "docmask_delta_by_position.png")
    fig.savefig(p1, dpi=150)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, smooth(a["overall"]), lw=2, label="arm A (no mask)")
    ax.plot(x, smooth(b["overall"]), lw=2, label="arm B (doc mask)")
    ax.set_xlabel("position within 1024-token block")
    ax.set_ylabel("val loss")
    ax.set_title(f"Loss by position within block — step {step}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(LOGS, "loss_by_position.png")
    fig.savefig(p2, dpi=150)

    fig, ax = plt.subplots(figsize=(9, 5))
    means, firsts, lasts = [], [], []
    for s in common:
        aa, bb = load(ARM_A, s), load(ARM_B, s)
        dd = bb["overall"] - aa["overall"]
        means.append(dd.mean()); firsts.append(dd[:64].mean()); lasts.append(dd[-64:].mean())
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(common, means, "o-", label="mean over block")
    ax.plot(common, firsts, "s--", label="first 64 positions")
    ax.plot(common, lasts, "^--", label="last 64 positions")
    ax.set_xlabel("training step")
    ax.set_ylabel("val loss delta (doc-mask − baseline)")
    ax.set_title("Document-masking penalty over training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(LOGS, "docmask_delta_over_training.png")
    fig.savefig(p3, dpi=150)

    print(f"\nfigures -> {p1}\n            {p2}\n            {p3}")


if __name__ == "__main__":
    main()