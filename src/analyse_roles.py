import os
import re
import json
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training_data/raw/Phase_2_LNER/Train")

ROLES = ["APP", "RESP", "WIT", "A.COUNSEL", "R.COUNSEL", "JUDGE"]
WINDOW = 60


def load(fold):
    with open(os.path.join(TRAIN, f"{fold}-text.json")) as f:
        docs = json.load(f)
    with open(os.path.join(TRAIN, f"{fold}-gold.json")) as f:
        gold = json.load(f)
    return docs, gold


def main():
    docs, gold = {}, {}
    for f in ("fold1", "fold2"):
        d, g = load(f)
        docs.update(d)
        gold.update(g)

    print("=" * 70)
    print("relative position in the document (0 = start, 1 = end)")
    print("=" * 70)
    pos = defaultdict(list)
    for did, ents in gold.items():
        n = len(docs[did])
        for e in ents:
            if e[2] in ROLES:
                pos[e[2]].append(e[0] / n)

    print(f"  {'label':<12} {'n':>5} {'p10':>6} {'median':>7} {'p90':>6} "
          f"{'% in first 10%':>15}")
    for lab in ROLES:
        a = np.array(pos[lab])
        if not len(a):
            continue
        print(f"  {lab:<12} {len(a):>5} {np.percentile(a,10):>6.2f} "
              f"{np.median(a):>7.2f} {np.percentile(a,90):>6.2f} "
              f"{100*(a < 0.1).mean():>14.1f}%")

    print("\n" + "=" * 70)
    print("characteristic context (words within 60 chars, by role)")
    print("=" * 70)
    ctx_before = defaultdict(Counter)
    ctx_after = defaultdict(Counter)
    for did, ents in gold.items():
        t = docs[did]
        for e in ents:
            if e[2] not in ROLES:
                continue
            b = t[max(0, e[0] - WINDOW):e[0]].lower()
            a = t[e[1]:e[1] + WINDOW].lower()
            ctx_before[e[2]].update(re.findall(r"[a-z]{3,}", b))
            ctx_after[e[2]].update(re.findall(r"[a-z]{3,}|\(\w+-?\d*\)", a))

    allb = Counter()
    for c in ctx_before.values():
        allb.update(c)
    alla = Counter()
    for c in ctx_after.values():
        alla.update(c)

    for lab in ROLES:
        n = len(pos[lab])
        if not n:
            continue
        def distinctive(c, total):
            return sorted(
                ((v / n) / max(1e-9, total[k] / sum(len(pos[l]) for l in ROLES)), k, v)
                for k, v in c.items() if v >= max(5, n * 0.05))[::-1][:6]
        print(f"\n  {lab} ({n} instances)")
        print("    before:", ", ".join(
            f"{k} ({v})" for _, k, v in distinctive(ctx_before[lab], allb)))
        print("    after: ", ", ".join(
            f"{k} ({v})" for _, k, v in distinctive(ctx_after[lab], alla)))

    print("\n" + "=" * 70)
    print("bracketed markers immediately after the entity")
    print("=" * 70)
    marker = re.compile(r"^\s*\(([A-Za-z]{1,3})\s*-?\s*\d*\)")
    hits = defaultdict(Counter)
    for did, ents in gold.items():
        t = docs[did]
        for e in ents:
            if e[2] not in ROLES:
                continue
            m = marker.match(t[e[1]:e[1] + 12])
            hits[e[2]][m.group(1).upper() if m else "-none-"] += 1

    print(f"  {'label':<12} {'n':>5}  markers")
    for lab in ROLES:
        c = hits[lab]
        n = sum(c.values())
        if not n:
            continue
        top = ", ".join(f"{k}:{v} ({100*v/n:.0f}%)"
                        for k, v in c.most_common(4))
        print(f"  {lab:<12} {n:>5}  {top}")

    print("\n" + "=" * 70)
    print("ambiguity of the surface string itself")
    print("=" * 70)
    by_text = defaultdict(Counter)
    for did, ents in gold.items():
        t = docs[did]
        for e in ents:
            if e[2] in ROLES:
                by_text[t[e[0]:e[1]].strip().lower()][e[2]] += 1

    amb = {k: v for k, v in by_text.items() if len(v) > 1}
    print(f"  {len(by_text):,} distinct strings, {len(amb):,} take more than "
          f"one role ({100*len(amb)/len(by_text):.1f}%)")
    inst_amb = sum(sum(v.values()) for v in amb.values())
    inst_all = sum(sum(v.values()) for v in by_text.values())
    print(f"  {inst_amb:,} of {inst_all:,} instances ({100*inst_amb/inst_all:.1f}%) "
          f"are of an ambiguous string")
    print("\n  most ambiguous:")
    for k, v in sorted(amb.items(), key=lambda kv: -sum(kv[1].values()))[:8]:
        print(f"    {k[:40]!r:<44} {dict(v)}")

    print("\n" + "=" * 70)
    print("within a single document, does one string ever take two roles?")
    print("=" * 70)
    inconsistent = 0
    total_strings = 0
    for did, ents in gold.items():
        t = docs[did]
        d = defaultdict(set)
        for e in ents:
            if e[2] in ROLES:
                d[t[e[0]:e[1]].strip().lower()].add(e[2])
        total_strings += len(d)
        inconsistent += sum(1 for v in d.values() if len(v) > 1)
    print(f"  {inconsistent} of {total_strings} document-string pairs "
          f"({100*inconsistent/total_strings:.1f}%) take more than one role")
    print("\n  if this is near zero, a document-level consistency constraint")
    print("  at decode time would be a cheap and legitimate gain")


if __name__ == "__main__":
    main()