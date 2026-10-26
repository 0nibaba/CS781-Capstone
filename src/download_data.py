"""
Download the Phase 2 L-NER data, splitting it to match the Phase 1 layout:

    training_data/raw/Phase_2_LNER/Train/   fold1/fold2 text + gold, labels.txt
    test_data/Phase_2_LNER/Test/            fold3-text.json

Expects HF_TOKEN in the environment:
    export HF_TOKEN=hf_...
"""

import os
import shutil
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "Exploration-Lab/CS-781-Capstone"
SUBDIR = "Phase_2_LNER"

STAGING = os.path.join(ROOT, ".cache/phase2_staging")
TRAIN_DST = os.path.join(ROOT, "training_data/raw", SUBDIR)
TEST_DST = os.path.join(ROOT, "test_data", SUBDIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache/huggingface"))

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set:  export HF_TOKEN=hf_...")

print(f"downloading {SUBDIR} from {REPO}...")
snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=[f"{SUBDIR}/**"],
    local_dir=STAGING,
    token=token,
)

src = os.path.join(STAGING, SUBDIR)
if not os.path.isdir(src):
    raise SystemExit(f"nothing downloaded at {src} — check the folder name "
                     f"on the repo page")

# Train/ -> training_data, Test/ -> test_data
for sub, dst_parent in (("Train", TRAIN_DST), ("Test", TEST_DST)):
    s = os.path.join(src, sub)
    if not os.path.isdir(s):
        print(f"warning: no {sub}/ in the download")
        continue
    d = os.path.join(dst_parent, sub)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(s, d)

shutil.rmtree(STAGING, ignore_errors=True)

print()
for label, base in (("training_data", TRAIN_DST), ("test_data", TEST_DST)):
    for dirpath, _, files in os.walk(base):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            print(f"  {label}/.../{os.path.relpath(p, base)}  "
                  f"{os.path.getsize(p)/1e6:.2f} MB")