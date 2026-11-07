import os
import math
import random
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from encoder_decoder import EncDecConfig, EncoderDecoder
from span_corruption import corrupt_sequence, EOS_ID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

TEXT = ("The High Court in revision overturned the Magistrate, stating that "
        "he acted beyond the scope of his authority. The primary question "
        "before this Court is whether the Magistrate erred in taking "
        "cognizance based on a police report filed under Section 173(2) of "
        "the Code.")


@torch.no_grad()
def generate(model, enc_idx, max_new_tokens=200, temperature=0.0,
             device="cpu", stop_at_eos=True):
    model.eval()
    enc_out = model.encode(enc_idx)
    dec = torch.tensor([[EOS_ID]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model.decode(dec, enc_out)[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(
                F.softmax(logits / temperature, dim=-1), 1)
        dec = torch.cat([dec, nxt], dim=1)
        if stop_at_eos and nxt.item() == EOS_ID:
            break
    return dec[0, 1:].tolist()


def train_overfit(tok, sent, device, steps=600, lr=1e-3):
    """Same tiny model as test 6 — memorises one corrupted paragraph."""
    torch.manual_seed(0)
    rng = random.Random(0)
    ids = tok.encode(TEXT).ids
    enc, din, dt = corrupt_sequence(ids, sent, p=0.15, mean_span=5, rng=rng)

    cfg = EncDecConfig(vocab_size=tok.get_vocab_size(), block_size=256,
                       n_encoder_layer=2, n_decoder_layer=2, n_head=4,
                       d_model=128, dropout=0.0)
    m = EncoderDecoder(cfg).to(device)
    opt = m.configure_optimizers(0.0, lr, device)

    e = torch.tensor([enc], dtype=torch.long, device=device)
    di = torch.tensor([din], dtype=torch.long, device=device)
    dtg = torch.tensor([dt], dtype=torch.long, device=device)

    m.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = m(e, di, dtg)
        loss.backward()
        opt.step()
    print(f"  overfit model trained, final loss {loss.item():.4f}")
    return m, enc, dt, ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(TOK)
    sent = [tok.token_to_id(f"<|unused{i}|>") for i in range(10)]
    d = lambda ids: tok.decode(ids, skip_special_tokens=False)

    print("--- training the overfit model ---")
    model, enc_train, dt_train, ids = train_overfit(tok, sent, device)

    print("\n--- A. mechanics: can it reproduce a memorised fill? ---")
    e = torch.tensor([enc_train], dtype=torch.long, device=device)
    out = generate(model, e, max_new_tokens=60, device=device)
    print(f"  expected : {d(dt_train)}")
    print(f"  generated: {d(out)}")
    ok = out[:len(dt_train)] == dt_train
    print(f"  exact reproduction: {ok}")
    starts_sentinel = len(out) > 0 and out[0] in sent
    print(f"  starts with a sentinel: {starts_sentinel}")
    ends_eos = EOS_ID in out
    print(f"  emits EOS: {ends_eos}")

    print("\n--- B. a sentinel at the END, with no right context ---")
    print("  (the training regime never puts a sentinel last, so this is the")
    print("   regime Phase 1 would actually use)")
    prompt_ids = tok.encode(TEXT).ids[:30]
    enc_tail = prompt_ids + [sent[0]]
    e2 = torch.tensor([enc_tail], dtype=torch.long, device=device)

    for temp in (0.0, 0.8):
        out2 = generate(model, e2, max_new_tokens=200, temperature=temp,
                        device=device)
        n_before_eos = out2.index(EOS_ID) if EOS_ID in out2 else len(out2)
        txt = d(out2)
        print(f"\n  temperature {temp}: {n_before_eos} tokens before EOS")
        print(f"    {txt[:300]}")

    print("\n--- C. with EOS suppressed, does it keep producing text? ---")
    print("  (Phase 1 needs ~166 tokens; training spans are 25)")
    out3 = generate(model, e2, max_new_tokens=200, temperature=0.8,
                    device=device, stop_at_eos=False)
    print(f"  {len(out3)} tokens generated")
    print(f"    {d(out3)[:400]}")

    print("\n--- verdict ---")
    print("  A failing  -> the training format or the decoder is wired wrong.")
    print("  A passing, B producing only ~span-length output -> the model will")
    print("    not generate 166 tokens from a terminal sentinel. Fix BEFORE")
    print("    training: include spans that run to the end of the sequence,")
    print("    and/or sample some much longer spans, in corrupt_sequence.")
    print("  Note this is a 2-layer model overfit on ONE paragraph, so output")
    print("  quality means nothing here — only the structural behaviour does.")


if __name__ == "__main__":
    main()