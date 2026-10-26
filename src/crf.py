"""
Linear-chain CRF for BIO tagging.

Motivation, measured rather than assumed: post-hoc gap bridging improved
macro-F1 monotonically (0.3076 -> 0.3497 -> 0.3525 at gaps 0, 2, 3), which
means BIO chains break mid-entity and patching them afterwards recovers real
entities. A CRF scores whole sequences instead of tokens independently, so it
should prevent the breaks rather than repair them.

Impossible transitions (O -> I-X, B-X -> I-Y) are hard-constrained to -1e4
rather than left to be learned. With 56 training documents there is not enough
signal to learn structural rules that are known a priori.

The forward algorithm and Viterbi are verified against brute-force enumeration
over all tag sequences for a small case — CRF implementations fail silently on
indexing errors, and a wrong partition function still trains to something.
"""

import torch
import torch.nn as nn

from bio_tagging import TAGS, ID2TAG

NEG = -1e4


def build_constraint_mask(tags=TAGS):
    """
    (n, n) bool: allowed[i][j] is True if tag j may follow tag i.
    Also returns start/end masks.
    """
    n = len(tags)
    allowed = torch.ones(n, n, dtype=torch.bool)
    start_ok = torch.ones(n, dtype=torch.bool)

    for i, ti in enumerate(tags):
        for j, tj in enumerate(tags):
            if tj == "O":
                continue
            pj, lj = tj.split("-", 1)
            if pj == "B":
                continue                      # B-X may follow anything
            # I-X may only follow B-X or I-X
            if ti == "O":
                allowed[i][j] = False
            else:
                pi, li = ti.split("-", 1)
                if li != lj:
                    allowed[i][j] = False

    for j, tj in enumerate(tags):             # a sequence cannot start with I-X
        if tj != "O" and tj.split("-", 1)[0] == "I":
            start_ok[j] = False

    return allowed, start_ok


class CRF(nn.Module):
    def __init__(self, n_tags=len(TAGS), constrain=True):
        super().__init__()
        self.n_tags = n_tags
        self.transitions = nn.Parameter(torch.randn(n_tags, n_tags) * 0.01)
        self.start = nn.Parameter(torch.randn(n_tags) * 0.01)
        self.end = nn.Parameter(torch.randn(n_tags) * 0.01)

        allowed, start_ok = build_constraint_mask()
        self.register_buffer("allowed", allowed, persistent=False)
        self.register_buffer("start_ok", start_ok, persistent=False)
        self.constrain = constrain
        if constrain:
            allowed, start_ok = build_constraint_mask()
            assert allowed.size(0) == n_tags, \
                "constraint mask built for a different tag set"
            self.register_buffer("allowed", allowed, persistent=False)
            self.register_buffer("start_ok", start_ok, persistent=False)

    def _trans(self):
        t = self.transitions
        s = self.start
        if self.constrain:
            t = t.masked_fill(~self.allowed, NEG)
            s = s.masked_fill(~self.start_ok, NEG)
        return t, s

    def _score(self, emissions, tags, mask):
        """Score of the given tag sequence. emissions (B,T,n), tags (B,T)."""
        B, T, _ = emissions.shape
        trans, start = self._trans()

        score = start[tags[:, 0]] + emissions[:, 0].gather(
            1, tags[:, :1]).squeeze(1)

        for t in range(1, T):
            s = trans[tags[:, t - 1], tags[:, t]] + \
                emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + s * mask[:, t]

        last = mask.sum(1).long() - 1
        last_tag = tags.gather(1, last.unsqueeze(1)).squeeze(1)
        return score + self.end[last_tag]

    def _logZ(self, emissions, mask):
        """Log partition function by the forward algorithm."""
        B, T, n = emissions.shape
        trans, start = self._trans()

        alpha = start.unsqueeze(0) + emissions[:, 0]          # (B, n)
        for t in range(1, T):
            # (B, n_prev, n_next)
            nxt = alpha.unsqueeze(2) + trans.unsqueeze(0) + \
                emissions[:, t].unsqueeze(1)
            nxt = torch.logsumexp(nxt, dim=1)                 # (B, n)
            m = mask[:, t].unsqueeze(1)
            alpha = nxt * m + alpha * (1 - m)                 # freeze if padded

        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def forward(self, emissions, tags, mask):
        """Negative log-likelihood, averaged over the batch."""
        mask = mask.float()
        return (self._logZ(emissions, mask) -
                self._score(emissions, tags, mask)).mean()

    @torch.no_grad()
    def viterbi(self, emissions, mask):
        """Best valid tag sequence per item. Returns a list of lists."""
        B, T, n = emissions.shape
        trans, start = self._trans()
        maskf = mask.float()

        score = start.unsqueeze(0) + emissions[:, 0]
        history = []

        for t in range(1, T):
            nxt = score.unsqueeze(2) + trans.unsqueeze(0) + \
                emissions[:, t].unsqueeze(1)
            best, idx = nxt.max(dim=1)
            m = maskf[:, t].unsqueeze(1)
            score = best * m + score * (1 - m)
            history.append(idx)

        score = score + self.end.unsqueeze(0)
        lengths = mask.long().sum(1)
        best_last = score.argmax(dim=1)

        out = []
        for b in range(B):
            L = int(lengths[b].item())
            tag = int(best_last[b].item())
            seq = [tag]
            for t in range(L - 2, -1, -1):
                tag = int(history[t][b, tag].item())
                seq.append(tag)
            out.append(seq[::-1])
        return out


# ---------------- verification ----------------

def _brute_force_logZ(crf, emissions):
    """Enumerate every tag sequence. Only tractable for tiny n and T."""
    import itertools
    B, T, n = emissions.shape
    trans, start = crf._trans()
    out = []
    for b in range(B):
        scores = []
        for seq in itertools.product(range(n), repeat=T):
            s = start[seq[0]] + emissions[b, 0, seq[0]]
            for t in range(1, T):
                s = s + trans[seq[t - 1], seq[t]] + emissions[b, t, seq[t]]
            scores.append(s + crf.end[seq[-1]])
        out.append(torch.logsumexp(torch.stack(scores), dim=0))
    return torch.stack(out)


def _brute_force_viterbi(crf, emissions):
    import itertools
    B, T, n = emissions.shape
    trans, start = crf._trans()
    out = []
    for b in range(B):
        best, best_seq = None, None
        for seq in itertools.product(range(n), repeat=T):
            s = start[seq[0]] + emissions[b, 0, seq[0]]
            for t in range(1, T):
                s = s + trans[seq[t - 1], seq[t]] + emissions[b, t, seq[t]]
            s = s + crf.end[seq[-1]]
            if best is None or s > best:
                best, best_seq = s, list(seq)
        out.append(best_seq)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)

    print("--- 1. constraint mask ---")
    allowed, start_ok = build_constraint_mask()
    n = len(TAGS)
    print(f"  {n} tags, {int(allowed.sum())} of {n*n} transitions allowed")
    print(f"  sequences may start with {int(start_ok.sum())} of {n} tags")
    o = TAGS.index("O")
    i_date = TAGS.index("I-DATE")
    b_date = TAGS.index("B-DATE")
    i_prec = TAGS.index("I-PREC")
    assert not allowed[o][i_date], "O -> I-DATE must be forbidden"
    assert allowed[b_date][i_date], "B-DATE -> I-DATE must be allowed"
    assert not allowed[b_date][i_prec], "B-DATE -> I-PREC must be forbidden"
    assert allowed[i_date][b_date], "I-DATE -> B-DATE must be allowed"
    assert not start_ok[i_date], "cannot start with I-DATE"
    print("  O->I forbidden, B-X->I-X allowed, B-X->I-Y forbidden: all correct\n")

    print("--- 2. forward algorithm vs brute force (4 tags, T=4) ---")
    small = CRF(n_tags=4, constrain=False)
    em = torch.randn(3, 4, 4)
    mask = torch.ones(3, 4)
    got = small._logZ(em, mask)
    want = _brute_force_logZ(small, em)
    print(f"  forward    {got.tolist()}")
    print(f"  brute      {want.tolist()}")
    d = (got - want).abs().max().item()
    print(f"  max diff   {d:.2e}")
    assert d < 1e-4, "partition function is wrong"
    print()

    print("--- 3. viterbi vs brute force ---")
    got = small.viterbi(em, mask.long())
    want = _brute_force_viterbi(small, em)
    for g, w in zip(got, want):
        print(f"  viterbi {g}  brute {w}  {'ok' if g == w else 'MISMATCH'}")
        assert g == w
    print()

    print("--- 4. logZ >= any single sequence score ---")
    tags = torch.randint(0, 4, (3, 4))
    sc = small._score(em, tags, mask)
    lz = small._logZ(em, mask)
    print(f"  scores {sc.tolist()}")
    print(f"  logZ   {lz.tolist()}")
    assert (lz >= sc - 1e-5).all(), "logZ below a sequence score"
    print("  ok\n")

    print("--- 5. padding is respected ---")
    crf = CRF(n_tags=4, constrain=False)
    em = torch.randn(2, 6, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    lz_pad = crf._logZ(em, mask.float())
    lz_short = crf._logZ(em[:1, :4], torch.ones(1, 4))
    print(f"  padded item logZ {lz_pad[0].item():.6f}")
    print(f"  same item unpadded {lz_short[0].item():.6f}")
    d = abs(lz_pad[0].item() - lz_short[0].item())
    print(f"  diff {d:.2e}")
    assert d < 1e-4, "padding changes the partition function"
    print("  ok\n")

    print("--- 6. constrained decoding never emits an invalid sequence ---")
    crf = CRF()
    em = torch.randn(8, 20, len(TAGS)) * 3
    seqs = crf.viterbi(em, torch.ones(8, 20, dtype=torch.long))
    bad = 0
    for s in seqs:
        for a, b in zip(s, s[1:]):
            if not allowed[a][b]:
                bad += 1
        if not start_ok[s[0]]:
            bad += 1
    print(f"  {bad} invalid transitions across 8 random sequences")
    assert bad == 0
    print("  ok")