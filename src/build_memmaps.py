import os
import json
import ijson
import hashlib
import argparse
import numpy as np
import time
import re
from tokenizers import Tokenizer


def clean_headers(text, strip_headers):
    if strip_headers and "PETITIONER:" in text:
        parts = re.split(r'(?i)JUDGMENT[:\s]|ORDER[:\s]', text, maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return text


def flush_buffer(buffer, filepath):
    if not buffer:
        return
    np.array(buffer, dtype=np.uint16).tofile(open(filepath, 'ab'))
    buffer.clear()


def encode_and_route(texts, is_short_list, tokenizer, eot_id,
                     main_buffer, short_buffer, stats):
    encodings = tokenizer.encode_batch(texts)
    for enc, is_short in zip(encodings, is_short_list):
        ids = enc.ids + [eot_id]
        if is_short:
            short_buffer.extend(ids)
            stats['tokens_short'] += len(ids)
            stats['docs_kept_short'] += 1
        else:
            main_buffer.extend(ids)
            stats['tokens_main'] += len(ids)
            stats['docs_kept_main'] += 1


def process_source(category, filename, data_dir, out_dir, tokenizer, eot_id,
                   heldout_set, seen_hashes, args, meta):
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {category}: {filepath} not found.")
        return

    main_bin = os.path.join(out_dir, f"{category}.bin")
    short_bin = os.path.join(out_dir, f"{category}_short.bin")
    open(main_bin, 'wb').close()
    open(short_bin, 'wb').close()

    stats = {
        'docs_kept_main': 0, 'docs_kept_short': 0,
        'docs_skipped_heldout': 0, 'docs_skipped_dupe': 0,
        'tokens_main': 0, 'tokens_short': 0,
    }

    main_buffer, short_buffer = [], []
    batch_texts, batch_is_short = [], []
    last_report = 0
    start = time.time()

    print(f"Processing {category}...", flush=True)
    with open(filepath, 'rb') as f:
        for item in ijson.items(f, 'item'):
            raw_text = item.get('text', '')
            if not raw_text:
                continue

            # hash RAW text with md5 — must match build_heldout.py exactly
            doc_hash = hashlib.md5(raw_text.encode('utf-8')).hexdigest()

            if doc_hash in heldout_set:
                stats['docs_skipped_heldout'] += 1
                continue
            if doc_hash in seen_hashes:
                stats['docs_skipped_dupe'] += 1
                continue
            seen_hashes.add(doc_hash)

            text = clean_headers(raw_text.replace('\n', ' ').strip(),
                                 args.strip_headers)
            if not text:
                continue

            batch_texts.append(text)
            batch_is_short.append(len(text.split()) < args.short_threshold)

            if len(batch_texts) >= args.batch_size:
                encode_and_route(batch_texts, batch_is_short, tokenizer, eot_id,
                                 main_buffer, short_buffer, stats)
                batch_texts, batch_is_short = [], []

                if len(main_buffer) >= args.flush_size:
                    flush_buffer(main_buffer, main_bin)
                if len(short_buffer) >= args.flush_size:
                    flush_buffer(short_buffer, short_bin)

                total = stats['docs_kept_main'] + stats['docs_kept_short']
                if total - last_report >= 50_000:
                    el = time.time() - start
                    print(f"  {total:,} docs | "
                          f"{stats['tokens_main'] + stats['tokens_short']:,} tokens | "
                          f"{el:.0f}s", flush=True)
                    last_report = total

    if batch_texts:
        encode_and_route(batch_texts, batch_is_short, tokenizer, eot_id,
                         main_buffer, short_buffer, stats)

    flush_buffer(main_buffer, main_bin)
    flush_buffer(short_buffer, short_bin)

    meta[category] = stats
    print(f"Finished {category} in {time.time() - start:.0f}s — "
          f"main {stats['tokens_main']:,} tok / short {stats['tokens_short']:,} tok | "
          f"skipped {stats['docs_skipped_dupe']:,} dupes, "
          f"{stats['docs_skipped_heldout']:,} held-out\n", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="training_data/raw/Phase_1_Pretraining/Train")
    p.add_argument("--out_dir", default="training_data/tokens")
    p.add_argument("--tokenizer_path", default="checkpoints/tokenizers/legal_32000.json")
    p.add_argument("--heldout_hashes", default="test_data/heldout_hashes.json")
    p.add_argument("--batch_size", type=int, default=5000)
    p.add_argument("--flush_size", type=int, default=1_000_000)
    p.add_argument("--short_threshold", type=int, default=200)
    p.add_argument("--strip_headers", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        raise ValueError("Tokenizer must define <|endoftext|>")

    # heldout_hashes.json is {source: [hash, ...]} — flatten it
    heldout_set = set()
    if os.path.exists(args.heldout_hashes):
        with open(args.heldout_hashes) as f:
            d = json.load(f)
        heldout_set = {h for hashes in d.values() for h in hashes} \
                      if isinstance(d, dict) else set(d)
    print(f"Loaded {len(heldout_set):,} held-out hashes.\n")

    seen_hashes = set()
    meta = {'config': {
        'tokenizer': args.tokenizer_path,
        'vocab_size': tokenizer.get_vocab_size(),
        'eot_id': eot_id,
        'short_threshold': args.short_threshold,
        'stripped_headers': args.strip_headers,
    }}

    sources = {
        'acts': 'train_acts.json',
        'high': 'train_high_courts.json',
        'supreme': 'train_supreme_court.json',
        'tribunal': 'train_tribunals.json',
    }

    for category, filename in sources.items():
        process_source(category, filename, args.data_dir, args.out_dir,
                       tokenizer, eot_id, heldout_set, seen_hashes, args, meta)

    skipped = sum(m['docs_skipped_heldout'] for k, m in meta.items() if k != 'config')
    dupes = sum(m['docs_skipped_dupe'] for k, m in meta.items() if k != 'config')
    tok_main = sum(m['tokens_main'] for k, m in meta.items() if k != 'config')
    tok_short = sum(m['tokens_short'] for k, m in meta.items() if k != 'config')

    meta['totals'] = {'tokens_main': tok_main, 'tokens_short': tok_short,
                      'heldout_skipped': skipped, 'dupes_skipped': dupes}

    with open(os.path.join(args.out_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"TOTAL: {tok_main:,} main + {tok_short:,} short = "
          f"{tok_main + tok_short:,} tokens")
    print(f"Skipped {dupes:,} duplicates, {skipped:,} held-out")

    assert skipped >= 0.9 * len(heldout_set), (
        f"FAIL: only {skipped} of {len(heldout_set)} held-out docs were skipped — "
        "hash mismatch, your held-out set is leaking into training")
    print("held-out separation verified")