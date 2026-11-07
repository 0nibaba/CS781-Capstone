import random
import math

EOS_ID = 0


def _sample_lengths(n_spans, mean_span, rng, sigma=0.7):
    mu = math.log(mean_span)
    return [max(1, int(rng.lognormvariate(mu, sigma))) for _ in range(n_spans)]


def corrupt_sequence(tokens, sentinel_ids, p=0.15, mean_span=25,
                     p_terminal=0.25, p_long_terminal=0.15,
                     long_span_range=(120, 200), rng=None):

    rng = rng or random
    n = len(tokens)

    if rng.random() < p_long_terminal:
        lo, hi = long_span_range
        span_len = rng.randint(lo, min(hi, max(lo, n // 3)))
        if n - span_len >= 1:
            cut = n - span_len
            enc = tokens[:cut] + [sentinel_ids[0]]
            dec_t = [sentinel_ids[0]] + tokens[cut:] + [EOS_ID]
            return enc, [EOS_ID] + dec_t[:-1], dec_t

    n_mask = int(n * p)
    n_spans = max(1, min(len(sentinel_ids), round(n_mask / mean_span)))

    lengths = _sample_lengths(n_spans, mean_span, rng)
    scale = n_mask / max(1, sum(lengths))
    lengths = [max(1, int(l * scale)) for l in lengths]
    total_span = sum(lengths)

    n_gap_tokens = n - total_span
    if n_gap_tokens < n_spans + 1:
        return list(tokens), [EOS_ID], [EOS_ID]

    terminal = rng.random() < p_terminal

    if terminal:
        cuts = sorted(rng.sample(range(1, n_gap_tokens), n_spans - 1)) \
               if n_spans > 1 else []
        bounds = [0] + cuts + [n_gap_tokens]
        gaps = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        gaps.append(0)
    else:
        cuts = sorted(rng.sample(range(1, n_gap_tokens), n_spans))
        gaps = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, n_spans)] \
               + [n_gap_tokens - cuts[-1]]

    enc, dec_t, pos = [], [], 0
    for i, (g, l) in enumerate(zip(gaps, lengths)):
        enc.extend(tokens[pos:pos + g])
        pos += g
        enc.append(sentinel_ids[i])
        dec_t.append(sentinel_ids[i])
        dec_t.extend(tokens[pos:pos + l])
        pos += l
    enc.extend(tokens[pos:])
    dec_t.append(EOS_ID)

    return enc, [EOS_ID] + dec_t[:-1], dec_t


if __name__ == "__main__":
    import os
    from collections import Counter
    from tokenizers import Tokenizer

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rng = random.Random(0)
    sent = list(range(90000, 90010))

    print("--- 1. structural checks ---")
    toks = list(range(100, 1124))
    enc, din, dt = corrupt_sequence(toks, sent, rng=rng)
    n_s = sum(1 for t in enc if t in sent)
    masked = len(toks) - (len(enc) - n_s)
    print(f"  original {len(toks)} | enc {len(enc)} | target {len(dt)}")
    print(f"  spans {n_s}, masked {masked} ({100*masked/len(toks):.1f}%)")

    order_e = [t for t in enc if t in sent]
    order_t = [t for t in dt if t in sent]
    assert order_e == order_t == sorted(order_e), "sentinel order wrong"
    c = Counter(t for t in enc if t not in sent) + \
        Counter(t for t in dt if t not in sent and t != EOS_ID)
    assert c == Counter(toks), "tokens lost or duplicated"
    assert din[0] == EOS_ID and din[1:] == dt[:-1], "shift wrong"
    print("  order, token conservation and shift all correct\n")

    print("--- 2. terminal spans appear at the stated rate ---")
    n_term, N = 0, 2000
    for _ in range(N):
        e, _, _ = corrupt_sequence(list(range(100, 1124)), sent, rng=rng)
        if e[-1] in sent:
            n_term += 1
    print(f"  {100*n_term/N:.1f}% of sequences end with a sentinel "
          f"(target {100*0.25:.0f}%)\n")

    print("--- 3. span-length distribution is heavy-tailed ---")
    lens = []
    for _ in range(2000):
        e, _, t = corrupt_sequence(list(range(100, 1124)), sent, rng=rng)
        cur = 0
        for tok in t:
            if tok in sent:
                if cur:
                    lens.append(cur)
                cur = 0
            elif tok != EOS_ID:
                cur += 1
        if cur:
            lens.append(cur)
    lens.sort()
    q = lambda f: lens[int(f * len(lens))]
    print(f"  span lengths: p10 {q(.1)}  median {q(.5)}  p90 {q(.9)}  "
          f"p99 {q(.99)}  max {lens[-1]}")
    print(f"  {100*sum(1 for l in lens if l > 60)/len(lens):.1f}% of spans "
          f"exceed 60 tokens")
    print("  (Phase 1 needs ~166 tokens from one sentinel, so a tail matters)\n")

    print("--- 4. readable example with a terminal span ---")
    tok = Tokenizer.from_file(
        os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json"))
    sent_real = [tok.token_to_id(f"<|unused{i}|>") for i in range(10)]
    text = ("The High Court in revision overturned the Magistrate, stating that "
            "he acted beyond the scope of his authority. The primary question "
            "before this Court is whether the Magistrate erred in taking "
            "cognizance based on a police report filed under Section 173(2) of "
            "the Code. It is well settled that the power to direct an "
            "investigation under Section 156(3) may be exercised only at the "
            "pre-cognizance stage, and not thereafter.")
    ids = tok.encode(text).ids
    for attempt in range(20):
        e, _, t = corrupt_sequence(ids, sent_real, mean_span=8,
                                   p_terminal=1.0, rng=rng)
        if e[-1] in sent_real:
            break
        
    print("--- 5. long-terminal mode ---")
    n_long, lens_long, N = 0, [], 2000
    for _ in range(N):
        e, _, t = corrupt_sequence(list(range(100, 1124)), sent, rng=rng)
        n_s = sum(1 for x in e if x in sent)
        if n_s == 1 and e[-1] in sent:
            n_long += 1
            lens_long.append(len(t) - 2)
    print(f"  {100*n_long/N:.1f}% single long terminal span "
          f"(target {100*0.15:.0f}%)")
    if lens_long:
        lens_long.sort()
        print(f"  their lengths: min {lens_long[0]}  median "
              f"{lens_long[len(lens_long)//2]}  max {lens_long[-1]}")
        print(f"  Phase 1 needs ~166 tokens — "
              f"{100*sum(1 for l in lens_long if l >= 166)/len(lens_long):.0f}% "
              f"of these reach it\n")

    
    print(f"  encoder input:\n    {tok.decode(e, skip_special_tokens=False)}")
    print(f"\n  decoder target:\n    {tok.decode(t, skip_special_tokens=False)}")
    print("\n  note the encoder input ENDS with a sentinel — this is the")
    print("  Phase 1 inference format, now present in training.")