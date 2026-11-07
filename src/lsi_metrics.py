import numpy as np


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def scores(prob, gold, thr):
    pred = prob >= thr
    tp = (pred & (gold == 1)).sum(0)
    fp = (pred & (gold == 0)).sum(0)
    fn = ((~pred) & (gold == 1)).sum(0)
    per = [prf(tp[i], fp[i], fn[i]) for i in range(gold.shape[1])]
    macro = float(np.mean([f for _, _, f in per]))
    micro = prf(tp.sum(), fp.sum(), fn.sum())[2]
    subset = float((pred == (gold == 1)).all(1).mean())
    return macro, micro, subset, per


def best_global(prob, gold, grid=None):
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.025)
    best = (-1.0, 0.5)
    for t in grid:
        m = scores(prob, gold, t)[0]
        if m > best[0]:
            best = (m, float(t))
    return best


def best_per_label(prob, gold, grid=None):
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.025)
    L = gold.shape[1]
    thr = np.full(L, 0.5)
    for i in range(L):
        g = gold[:, i]
        p = prob[:, i]
        best = (-1.0, 0.5)
        for t in grid:
            pr = p >= t
            f = prf((pr & (g == 1)).sum(), (pr & (g == 0)).sum(),
                    ((~pr) & (g == 1)).sum())[2]
            if f > best[0]:
                best = (f, float(t))
        thr[i] = best[1]
    return scores(prob, gold, thr)[0], thr