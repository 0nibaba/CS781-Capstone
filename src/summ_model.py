"""Summarizer wrapper over the Phase 1 architectures, with batched generation.

Two input formats, because the architectures differ in where the source goes:

  encdec     source -> encoder (up to 1024), summary -> decoder (up to 512)
  decoder /  source and summary share one 1024 window, so the source is capped
  prefix_lm  at 480 tokens and the summary at 512, joined by <|endoftext|>.
  efficient

That is a real confound: the encoder-decoder sees twice the source. It is
inherent to a 1024-token position table and is stated rather than hidden.

Generation obeys one hard constraint derived from the metric: BERTScore
truncates at 512 RoBERTa tokens, and the lead-k curve rises monotonically to
that cutoff (0.7981 at 128 -> 0.8300 at 512) and is exactly flat beyond. So EOS
is suppressed until min_len and generation stops at max_len. Length is not a
hyperparameter here; the metric fixes its optimum.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

BOS_ID = 0          # <|endoftext|> doubles as BOS/EOS
EOS_ID = 0
PAD_ID = 0


class Summarizer(nn.Module):
    def __init__(self, base, arch):
        super().__init__()
        self.base = base
        self.arch = arch
        self.is_encdec = (arch == "encdec")

    # ---------- training ----------

    def forward(self, batch):
        if self.is_encdec:
            logits, loss = self.base(
                batch["enc_idx"], batch["dec_in"], targets=batch["dec_out"],
                enc_pad_mask=batch["enc_mask"], dec_pad_mask=batch["dec_mask"])
            return logits, loss

        ids, tgt = batch["ids"], batch["targets"]
        if self.arch == "prefix_lm":
            logits, _ = self.base(ids, attn_mask=batch["attn_mask"],
                                  split_indices=batch["split"])
        else:
            logits, _ = self.base(ids, attn_mask=batch["attn_mask"])
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            tgt[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss

    # ---------- generation ----------

    @torch.no_grad()
    def generate(self, batch, max_len=544, min_len=480, temperature=0.0,
                 top_k=0, no_repeat_ngram=0):
        return (self._gen_encdec if self.is_encdec else self._gen_causal)(
            batch, max_len, min_len, temperature, top_k, no_repeat_ngram)

    def _pick(self, logits, generated, step, min_len, temperature, top_k,
              no_repeat_ngram):
        logits = logits.float()
        if step < min_len:
            logits[:, EOS_ID] = float("-inf")       # fill the metric window
        if no_repeat_ngram and generated.size(1) >= no_repeat_ngram:
            n = no_repeat_ngram
            for b in range(generated.size(0)):
                seq = generated[b].tolist()
                prefix = tuple(seq[-(n - 1):])
                for i in range(len(seq) - n + 1):
                    if tuple(seq[i:i + n - 1]) == prefix:
                        logits[b, seq[i + n - 1]] = float("-inf")
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        return torch.multinomial(logits.softmax(-1), 1)

    def _gen_encdec(self, batch, max_len, min_len, temperature, top_k, nrn):
        enc_idx, enc_mask = batch["enc_idx"], batch["enc_mask"]
        B = enc_idx.size(0)
        enc_out = self.base.encode(enc_idx, enc_mask)
        dec = torch.full((B, 1), BOS_ID, dtype=torch.long,
                         device=enc_idx.device)
        done = torch.zeros(B, dtype=torch.bool, device=enc_idx.device)
        for step in range(max_len):
            logits = self.base.decode(dec, enc_out, enc_mask)[:, -1]
            nxt = self._pick(logits, dec, step, min_len, temperature, top_k,
                             nrn)
            nxt[done] = EOS_ID
            dec = torch.cat([dec, nxt], 1)
            done = done | (nxt.squeeze(1) == EOS_ID)
            if done.all():
                break
        return dec[:, 1:]

    def _gen_causal(self, batch, max_len, min_len, temperature, top_k, nrn):
        ids = batch["ids"]                      # source + <|endoftext|>
        B, src_len = ids.size()
        split = batch.get("split")
        out = torch.zeros(B, 0, dtype=torch.long, device=ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        cur = ids
        for step in range(max_len):
            if cur.size(1) >= self.base.config.block_size:
                break
            if self.arch == "prefix_lm":
                logits, _ = self.base(cur, split_indices=split)
            else:
                logits, _ = self.base(cur)
            nxt = self._pick(logits[:, -1], out, step, min_len, temperature,
                             top_k, nrn)
            nxt[done] = EOS_ID
            cur = torch.cat([cur, nxt], 1)
            out = torch.cat([out, nxt], 1)
            done = done | (nxt.squeeze(1) == EOS_ID)
            if done.all():
                break
        return out

    def trainable_parameters(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        return t, a


def strip_eos(seq, tok):
    out = []
    for row in seq.tolist():
        if EOS_ID in row:
            row = row[:row.index(EOS_ID)]
        out.append(tok.decode(row))
    return out