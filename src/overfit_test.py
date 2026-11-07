import torch
import argparse
from tokenizers import Tokenizer
from decoder_only_transformer import GPT, ModelConfig


PARAGRAPH = (
    "The High Court in revision overturned the Magistrate, stating that he acted "
    "beyond the scope of his authority. The primary question before this Court is "
    "whether the Magistrate erred in taking cognizance based on a police report "
    "filed under Section 173(2) of the Code. It is well settled that the power to "
    "direct an investigation under Section 156(3) may be exercised only at the "
    "pre-cognizance stage, and not thereafter."
)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(PARAGRAPH).ids
    print(f"paragraph -> {len(ids)} tokens "
          f"({len(PARAGRAPH.split())} words, "
          f"{len(ids)/len(PARAGRAPH.split()):.3f} tok/word)")

    cfg = ModelConfig(
        vocab_size=tok.get_vocab_size(),
        block_size=256,
        n_layer=2,
        n_head=4,
        d_model=128,
        dropout=0.0,
    )
    model = GPT(cfg).to(device)
    print(f"tiny model: {sum(p.numel() for p in model.parameters()):,} params")

    opt = model.configure_optimizers(weight_decay=0.0,
                                     learning_rate=args.lr,
                                     device_type=device)

    data = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    x, y = data[:, :-1], data[:, 1:]

    model.train()
    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    final = loss.item()

    model.eval()
    prompt_len = 5
    out = model.generate(x[:, :prompt_len], max_new_tokens=len(ids) - prompt_len,
                         temperature=1e-8)
    generated = tok.decode(out[0].tolist())

    print("\n--- greedy reconstruction ---")
    print(generated[:400])

    print("\n--- verdict ---")
    if final < 0.1:
        print(f"PASS  final loss {final:.4f} — model can memorize, wiring is sound")
    elif final < 1.0:
        print(f"MARGINAL  final loss {final:.4f} — try more steps or a higher lr")
    else:
        print(f"FAIL  final loss {final:.4f} — likely a wiring bug.")
        print("  loss stuck near ln(vocab) -> gradients not reaching the model")
        print("  loss plateaued mid-range  -> check the target shift (y = x[1:])")
        print("  loss ~0 within a few steps -> targets not shifted; learning identity")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="checkpoints/tokenizers/legal_32000.json")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    main(args)