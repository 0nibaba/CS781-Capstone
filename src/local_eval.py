import os
import csv
import json
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
from decoder_only_transformer import GPT, ModelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FIELDS = [
    "checkpoint", "config", "n", "bert_f1", "bert_p", "bert_r",
    "rouge_l", "bleu_4", "mean_word_count", "mean_rep_rate", "pct_short",
    "acts_bert_f1", "high_bert_f1", "supreme_bert_f1", "tribunal_bert_f1",
]


# ---------- diagnostics ----------

def distinct_ngram_ratio(text, n=4):
    w = text.split()
    if len(w) < n:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def truncate_words(text, k=128):
    return " ".join(text.split()[:k])


# ---------- sampling ----------

def apply_repetition_penalty(logits, generated, penalty):
    """Divide logits of already-generated tokens (multiply if negative)."""
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    for b in range(logits.size(0)):
        uniq = torch.unique(generated[b])
        vals = logits[b, uniq]
        logits[b, uniq] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    """Mask logits outside the top-k / nucleus set with -inf."""
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)

        remove = cumprobs - probs > top_p          # keep the token that crosses p
        remove[..., 0] = False                     # always keep the top token

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


@torch.no_grad()
def generate(model, idx, attn_mask, max_new_tokens, cfg, eos_id, device):
    """
    idx: (B, T) left-padded prompt token ids
    Returns (B, max_new_tokens) of generated ids.
    Assumes model(idx, attn_mask) -> logits (B, T, vocab).
    """
    model.eval()
    B = idx.size(0)
    out = torch.empty((B, 0), dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    block = cfg["block_size"]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block:]
        mask_cond = attn_mask[:, -block:] if attn_mask is not None else None

        try:
            logits = model(idx_cond, mask_cond) if mask_cond is not None else model(idx_cond)
        except TypeError:
            logits = model(idx_cond)

        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[:, -1, :].float()

        logits = apply_repetition_penalty(logits, out, cfg["repetition_penalty"])

        if cfg["suppress_eos"] and eos_id is not None:
            logits[:, eos_id] = float("-inf")

        if cfg["temperature"] <= 0:                       # greedy
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / cfg["temperature"]
            logits = top_k_top_p_filter(logits, cfg["top_k"], cfg["top_p"])
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)

        if eos_id is not None:
            nxt[finished] = eos_id
            finished |= nxt.squeeze(-1).eq(eos_id)

        out = torch.cat([out, nxt], dim=1)
        idx = torch.cat([idx, nxt], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat(
                [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=device)], dim=1)

        if finished.all():
            break

    return out


# ---------- prediction ----------

def generate_predictions(model, tokenizer, eval_set, cfg, device, eos_id):
    ids = list(eval_set.keys())
    preds, diags = {}, {}
    bs = cfg["batch_size"]

    for i in tqdm(range(0, len(ids), bs), desc="generating"):
        b_ids = ids[i:i + bs]
        b_prompts = [eval_set[k]["prompt"] for k in b_ids]

        encs = [tokenizer.encode(p).ids for p in b_prompts]
        # truncate from the LEFT — the end of the prompt is what matters
        limit = cfg["block_size"] - cfg["max_new_tokens"]
        encs = [e[-limit:] for e in encs]

        width = max(len(e) for e in encs)
        pad_id = eos_id if eos_id is not None else 0
        padded = [[pad_id] * (width - len(e)) + e for e in encs]      # LEFT pad
        mask = [[0] * (width - len(e)) + [1] * len(e) for e in encs]

        x = torch.tensor(padded, dtype=torch.long, device=device)
        m = torch.tensor(mask, dtype=torch.long, device=device)

        gen = generate(model, x, m, cfg["max_new_tokens"], cfg, eos_id, device)

        for j, k in enumerate(b_ids):
            seq = gen[j].tolist()
            if eos_id is not None and eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            text = tokenizer.decode(seq)

            raw_wc = len(text.split())
            trunc = truncate_words(text, 128)
            preds[k] = trunc
            diags[k] = {
                "word_count": len(trunc.split()),
                "raw_word_count": raw_wc,
                "repetition_4gram": distinct_ngram_ratio(trunc),
                "short": raw_wc < 128,
            }

    return preds, diags


# ---------- scoring ----------

def evaluate(preds, eval_set, diags, scorer):
    ids = list(preds.keys())
    cands = [preds[i] for i in ids]
    refs = [eval_set[i]["reference"] for i in ids]
    srcs = [eval_set[i].get("source", "unknown") for i in ids]

    print("computing BERTScore...")
    P, R, F1 = scorer.score(cands, refs)

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    rows = []
    for n, (c, r, s) in enumerate(zip(cands, refs, srcs)):
        rows.append({
            "id": ids[n], "source": s,
            "bert_f1": F1[n].item(), "bert_p": P[n].item(), "bert_r": R[n].item(),
            "rouge_l": rs.score(r, c)["rougeL"].fmeasure,
            "bleu_4": sentence_bleu([r.split()], c.split(), smoothing_function=smooth),
            **diags[ids[n]],
        })
    return rows


def aggregate_and_log(rows, checkpoint, config_str, log_csv="logs/eval_metrics.csv"):
    def mean(key, subset=None):
        vals = [r[key] for r in (subset or rows)]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {f: "" for f in CSV_FIELDS}
    summary.update({
        "checkpoint": checkpoint, "config": config_str, "n": len(rows),
        "bert_f1": round(mean("bert_f1"), 5),
        "bert_p": round(mean("bert_p"), 5),
        "bert_r": round(mean("bert_r"), 5),
        "rouge_l": round(mean("rouge_l"), 5),
        "bleu_4": round(mean("bleu_4"), 5),
        "mean_word_count": round(mean("word_count"), 1),
        "mean_rep_rate": round(mean("repetition_4gram"), 4),
        "pct_short": round(100 * sum(r["short"] for r in rows) / len(rows), 1),
    })

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for src, sub in by_src.items():
        key = f"{src}_bert_f1"
        if key in summary:
            summary[key] = round(mean("bert_f1", sub), 5)

    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    new = not os.path.isfile(log_csv)
    with open(log_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(summary)

    print(f"\nBERTScore F1 {summary['bert_f1']:.4f}  "
          f"(P {summary['bert_p']:.4f} / R {summary['bert_r']:.4f})")
    print(f"ROUGE-L {summary['rouge_l']:.4f}   BLEU-4 {summary['bleu_4']:.4f}")
    print(f"mean words {summary['mean_word_count']}  "
          f"distinct-4gram {summary['mean_rep_rate']}  "
          f"short {summary['pct_short']}%")
    for src in sorted(by_src):
        k = f"{src}_bert_f1"
        if summary.get(k) != "":
            print(f"  {src:9s} {summary[k]:.4f}")
    return summary


# ---------- entry ----------

if __name__ == "__main__":
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--eval_set", default=os.path.join(ROOT, "test_data/eval_set.json"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json"))
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--rep_penalty", type=float, default=1.1)
    p.add_argument("--allow_eos", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    with open(args.eval_set) as f:
        eval_set = json.load(f)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"loaded step {ck['step']}, val_loss {ck['best_val_loss']:.4f}")

    cfg = {
        "temperature": args.temp, "top_p": args.top_p, "top_k": args.top_k,
        "repetition_penalty": args.rep_penalty, "suppress_eos": not args.allow_eos,
        "batch_size": args.batch_size, "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
    }

    preds, diags = generate_predictions(model, tokenizer, eval_set, cfg, device, eos_id)

    scorer = BERTScorer(lang="en", rescale_with_baseline=False)   # match Codabench default
    rows = evaluate(preds, eval_set, diags, scorer)

    cfg_str = (f"T{args.temp}_p{args.top_p}_k{args.top_k}"
               f"_rp{args.rep_penalty}_eos{'on' if args.allow_eos else 'off'}")
    aggregate_and_log(rows, args.checkpoint, cfg_str)