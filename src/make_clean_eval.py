import os
import json
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD = 0.5

ev = json.load(open(os.path.join(ROOT, "test_data/eval_set.json")))
con = json.load(open(os.path.join(ROOT, "logs/contamination.json")))
scores = con["scores"]

clean = {k: v for k, v in ev.items() if scores.get(k, 0.0) < THRESHOLD}

out = os.path.join(ROOT, "test_data/eval_set_clean.json")
json.dump(clean, open(out, "w"), indent=2)

print(f"threshold: best-match jaccard < {THRESHOLD}")
print(f"kept {len(clean)} of {len(ev)} examples\n")
print(f"{'source':<10} {'original':>9} {'clean':>7} {'dropped':>8}")
print("-" * 38)
orig = Counter(v.get("source", "?") for v in ev.values())
keep = Counter(v.get("source", "?") for v in clean.values())
for s in sorted(orig):
    print(f"{s:<10} {orig[s]:>9} {keep[s]:>7} {orig[s]-keep[s]:>8}")
print(f"\nsaved -> {out}")
print("NOTE: the clean set is unbalanced across sources, so compare")
print("per-source numbers rather than only the aggregate.")