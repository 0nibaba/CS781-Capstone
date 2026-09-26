import os
import json
import zipfile
import argparse
import torch
from tqdm import tqdm
from tokenizers import Tokenizer

from decoder_only_transformer import GPT, ModelConfig
from local_eval import generate, truncate_words

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_submission(model, tokenizer, test_data, cfg, device, eos_id):
    ids = list(test_data.keys())
    preds = {}
    word_counts = []
    bs = cfg["batch_size"]

    for i in tqdm(range(0, len(ids), bs), desc="generating"):
        b_ids = ids[i:i + bs]
        b_prompts = [test_data[k]["text"] for k in b_ids]

        encs = [tokenizer.encode(p).ids for p in b_prompts]
        limit = cfg["block_size"] - cfg["max_new_tokens"]
        encs = [e[-limit:] for e in encs]

        width = max(len(e) for e in encs)
        pad_id = eos_id if eos_id is not None else 0
        padded = [[pad_id] * (width - len(e)) + e for e in encs]
        mask = [[0] * (width - len(e)) + [1] * len(e) for e in encs]

        x = torch.tensor(padded, dtype=torch.long, device=device)
        m = torch.tensor(mask, dtype=torch.long, device=device)

        gen = generate(model, x, m, cfg["max_new_tokens"], cfg, eos_id, device)

        for j, k in enumerate(b_ids):
            seq = gen[j].tolist()
            if eos_id is not None and eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            trunc = truncate_words(tokenizer.decode(seq), 128)
            preds[k] = {"text": trunc}          # submission requires this nesting
            word_counts.append(len(trunc.split()))

    return preds, word_counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_file",
                   default=os.path.join(ROOT, "test_data/Phase_1_Pretraining/Test/test.json"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer",
                   default=os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json"))
    p.add_argument("--tag", default="sub1", help="names the output directory")

    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--rep_penalty", type=float, default=1.1)
    p.add_argument("--allow_eos", action="store_true")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    with open(args.test_file) as f:
        test_data = json.load(f)
    print(f"{len(test_data)} test prompts")

    # prompt-length check: left truncation silently drops the START of a prompt
    lens = sorted(len(tokenizer.encode(v["text"]).ids) for v in test_data.values())
    limit = args.block_size - args.max_new_tokens
    n_over = sum(1 for l in lens if l > limit)
    print(f"prompt tokens: min {lens[0]}  median {lens[len(lens)//2]}  max {lens[-1]}"
          f"  (limit {limit})")
    if n_over:
        print(f"WARNING: {n_over} prompts exceed the limit and will be left-truncated")

    print(f"loading {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"  step {ck['step']}, val_loss {ck['best_val_loss']:.4f}")

    cfg = {
        "temperature": args.temp, "top_p": args.top_p, "top_k": args.top_k,
        "repetition_penalty": args.rep_penalty, "suppress_eos": not args.allow_eos,
        "batch_size": args.batch_size, "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
    }

    preds, wc = generate_submission(model, tokenizer, test_data, cfg, device, eos_id)

    print("\nvalidating...")
    missing = set(test_data) - set(preds)
    extra = set(preds) - set(test_data)
    assert not missing, f"FATAL: {len(missing)} test IDs missing from predictions"
    assert not extra, f"FATAL: {len(extra)} unknown IDs in predictions"
    assert all(len(v["text"].split()) <= 128 for v in preds.values()), \
        "FATAL: a generation exceeds 128 words"

    short = sum(1 for c in wc if c < 128)
    print(f"  ids: all {len(preds)} present, none extra")
    print(f"  mean words {sum(wc)/len(wc):.1f} / 128")
    print(f"  short: {short} ({100*short/len(wc):.1f}%)")
    if short > 0.1 * len(wc):
        print("  WARNING: >10% short — BERTScore recall will suffer")

    out_dir = os.path.join(ROOT, "submissions", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "prediction.json")
    zip_path = os.path.join(out_dir, "submission.zip")

    with open(pred_path, "w") as f:
        json.dump(preds, f, ensure_ascii=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")

    # verify the archive layout is what Codabench expects
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    assert names == ["prediction.json"], f"FATAL: bad archive layout {names}"

    cfg_str = (f"T{args.temp} p{args.top_p} k{args.top_k} rp{args.rep_penalty} "
               f"eos{'on' if args.allow_eos else 'off'} seed{args.seed}")
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\nstep: {ck['step']}\n"
                f"val_loss: {ck['best_val_loss']}\ndecoding: {cfg_str}\n"
                f"mean_words: {sum(wc)/len(wc):.1f}\nshort: {short}\n")

    print(f"\n-> {zip_path}")
    print(f"   json kept at {pred_path}")