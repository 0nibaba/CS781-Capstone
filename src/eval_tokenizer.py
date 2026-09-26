import os
import json
import argparse
from tokenizers import Tokenizer
import pandas as pd
from transformers import GPT2TokenizerFast

# from logs/corpus_profile.csv
WORD_PROFILE = {
    'acts':       119_638_300,
    'high':     1_684_692_412,
    'supreme':    210_205_151,
    'tribunal': 1_054_788_511,
}

D_MODEL = 768   # for the embedding-cost column


def score_tokenizer(tokenizer, name, vocab_size, eval_data, is_hf=False):
    row = {"name": name, "vocab_size": vocab_size}
    row['emb_params'] = f"{(vocab_size * D_MODEL) / 1e6:.1f}M"

    total_words = total_tokens = 0
    implied = 0.0

    for source, texts in eval_data.items():
        src_words = src_tokens = 0
        for text in texts:
            src_words += len(text.split())
            ids = tokenizer.encode(text) if is_hf else tokenizer.encode(text).ids
            src_tokens += len(ids)

        fertility = src_tokens / max(1, src_words)
        row[source] = fertility

        total_words += src_words
        total_tokens += src_tokens
        if source in WORD_PROFILE:
            implied += fertility * WORD_PROFILE[source]

    row['overall'] = total_tokens / max(1, total_words)
    row['implied_B'] = implied / 1e9
    return row

def save_results(results, path="logs/tokenizer_eval.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"\nsaved to {path}")

def print_table(results):
    hdr = (f"{'tokenizer':<16} | {'acts':<6} | {'high':<6} | {'supreme':<7} | "
           f"{'tribunal':<8} | {'overall':<7} | {'corpus':<7} | emb@{D_MODEL}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<16} | {r.get('acts', 0):.3f}  | {r.get('high', 0):.3f}  | "
              f"{r.get('supreme', 0):.3f}   | {r.get('tribunal', 0):.3f}    | "
              f"{r['overall']:.3f}   | {r['implied_B']:.2f}B   | {r['emb_params']}")


def validation_checks(tok):
    print("\n--- round-trip: does the continuation keep its leading space? ---")
    prompt = ("...writ petition entails with dismissal on grounds of laches. It is")
    cont   = " settled law that delay defeats equity."
    p_ids = tok.encode(prompt).ids
    f_ids = tok.encode(prompt + cont).ids

    stable = f_ids[:len(p_ids)] == p_ids
    print(f"prefix stable under concatenation: {stable}")
    if not stable:
        print("  WARNING: tokenization is not stable — slicing generated tokens will misalign")

    gen = f_ids[len(p_ids):]
    print("decoded continuation:", repr(tok.decode(gen)))
    print("(should begin with a leading space)")

    print("\n--- sanity: citation- and statute-heavy passage ---")
    crpc = ("Section 156(3) of the Code of Criminal Procedure empowers a "
            "Magistrate to direct an investigation. See AIR 1984 SC 1020 and "
            "(1997) 2 SCC 627.")
    print("tokens:", tok.encode(crpc).tokens)


def main(model_dir, eval_json, check_name):
    with open(eval_json) as f:
        eval_data = json.load(f)

    results = []

    gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
    results.append(score_tokenizer(gpt2, "gpt2", 50257, eval_data, is_hf=True))

    for file in sorted(os.listdir(model_dir)):
        if not file.endswith('.json'):
            continue
        tok = Tokenizer.from_file(os.path.join(model_dir, file))
        results.append(score_tokenizer(tok, file[:-5], tok.get_vocab_size(), eval_data))

    print_table(results)
    save_results(results)
    print("\nnote: emb params are the token table only (x2 if weights are untied)")

    path = os.path.join(model_dir, check_name)
    if os.path.exists(path):
        validation_checks(Tokenizer.from_file(path))
    else:
        print(f"\nskipping checks — {check_name} not found in {model_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="checkpoints/tokenizers")
    p.add_argument("--eval_data", default="test_data/eval_sample.json")
    p.add_argument("--check", default="legal_ds_32000.json",
                   help="which tokenizer to run the validation checks on")
    args = p.parse_args()
    main(args.model_dir, args.eval_data, args.check)